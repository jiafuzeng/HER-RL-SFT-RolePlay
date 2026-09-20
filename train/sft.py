#!/usr/bin/env python3
"""Unsloth SFT：用 Qwen3.6-27B 训角色扮演（sft_roleplay.jsonl）。

Qwen3.6-27B 是混合 Gated DeltaNet + 注意力的稠密 27B（约 56GB bf16 LoRA，A800 80GB 可跑）。
该系列不建议 QLoRA 4bit。需要 transformers v5 + 最新 unsloth。

    pip install -U unsloth unsloth_zoo bitsandbytes datasets trl peft accelerate pillow

    python train/sft.py --dry_run
    python train/sft.py
    python train/sft.py --max_steps 20 --output_dir /root/autodl-tmp/HER-train-out/sft-smoke
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTODL_TMP = Path("/root/autodl-tmp")
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "train_data" / "final_split" / "sft_roleplay.jsonl"
DEFAULT_OUTPUT_DIR = AUTODL_TMP / "HER-train-out" / "qwen36-27b-sft"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"

ASSISTANT_HEADER = "<|im_start|>assistant\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def _setup_cache_env() -> None:
    if AUTODL_TMP.is_dir():
        hf_home = AUTODL_TMP / "hf-home"
        tmp = AUTODL_TMP / "tmp"
        hf_home.mkdir(parents=True, exist_ok=True)
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
        os.environ.setdefault("TMPDIR", str(tmp))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def unwrap_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def ensure_user_turn(messages: list[dict]) -> list[dict]:
    """Qwen3.6 chat template 要求至少一条 user，否则会 raise。"""
    if any(m.get("role") == "user" for m in messages):
        return messages
    out = list(messages)
    idx = 1 if out and out[0].get("role") == "system" else 0
    out.insert(idx, {"role": "user", "content": "===Conversation Start==="})
    return out


def apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    messages = ensure_user_turn(messages)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def completion_char_start(text: str) -> int:
    pos = text.rfind(ASSISTANT_HEADER)
    if pos < 0:
        pos = text.rfind("<|im_start|>assistant")
        if pos < 0:
            return 0
        nl = text.find("\n", pos)
        start = nl + 1 if nl >= 0 else pos
    else:
        start = pos + len(ASSISTANT_HEADER)
    if text.startswith(EMPTY_THINK, start):
        start += len(EMPTY_THINK)
    return start


def trim_messages(messages: list[dict], max_chars: int) -> list[dict]:
    if not messages or max_chars <= 0:
        return messages
    total = sum(len(m.get("content") or "") for m in messages)
    if total <= max_chars:
        return messages
    sys_msg = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg is not None else list(messages)
    if not body:
        content = sys_msg["content"] if sys_msg else ""
        if sys_msg and len(content) > max_chars:
            return [{"role": "system", "content": content[:max_chars]}]
        return messages
    kept_rev: list[dict] = []
    used = len(sys_msg.get("content") or "") if sys_msg else 0
    for msg in reversed(body):
        extra = len(msg.get("content") or "")
        if kept_rev and used + extra > max_chars:
            break
        kept_rev.append(msg)
        used += extra
    kept = list(reversed(kept_rev))
    return ([sys_msg] + kept) if sys_msg is not None else kept


def tokenize_last_assistant(example: dict, tokenizer, max_seq_length: int, max_chars: int) -> dict:
    messages = example.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return {"input_ids": [], "attention_mask": [], "labels": []}
    messages = trim_messages(messages, max_chars)
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    start = completion_char_start(text)
    prefix = text[:start]
    full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    n_mask = min(len(prefix_ids), max(len(full_ids) - 1, 0))
    if n_mask and full_ids[:n_mask] != prefix_ids[:n_mask]:
        n_mask = 0
        limit = min(len(prefix_ids), len(full_ids) - 1)
        while n_mask < limit and full_ids[n_mask] == prefix_ids[n_mask]:
            n_mask += 1
    if len(full_ids) > max_seq_length:
        overflow = len(full_ids) - max_seq_length
        full_ids = full_ids[overflow:]
        n_mask = max(0, n_mask - overflow)
    n_mask = min(n_mask, max(len(full_ids) - 1, 0))
    labels = [-100] * n_mask + full_ids[n_mask:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def tokenize_all_assistant_text(example: dict, tokenizer, max_chars: int) -> dict:
    messages = example.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return {"text": ""}
    messages = trim_messages(messages, max_chars)
    return {"text": apply_chat_template(tokenizer, messages, add_generation_prompt=False)}


def load_jsonl_dataset(path: Path, max_samples: int | None):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    if "messages" in ds.column_names:
        ds = ds.filter(
            lambda row: bool(row.get("messages"))
            and row["messages"][-1].get("role") == "assistant"
        )
    if len(ds) == 0:
        raise FileNotFoundError(f"空数据集: {path}")
    return ds


def load_qwen36(model_name: str, max_seq_length: int, load_in_4bit: bool, load_in_16bit: bool):
    """优先 FastModel（VLM / 混合架构），失败则 FastLanguageModel。"""
    load_kwargs = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "load_in_16bit": load_in_16bit and not load_in_4bit,
        "load_in_8bit": False,
        "full_finetuning": False,
        "dtype": None,
        "auto_model": True,
        "fast_inference": False,
    }
    try:
        from unsloth import FastModel

        model, processor = FastModel.from_pretrained(
            **_filter_kwargs(FastModel.from_pretrained, load_kwargs)
        )
        return model, unwrap_tokenizer(processor), FastModel.get_peft_model
    except Exception as exc:
        print(f"FastModel 加载失败，改用 FastLanguageModel: {exc}")
        from unsloth import FastLanguageModel

        model, processor = FastLanguageModel.from_pretrained(
            **_filter_kwargs(FastLanguageModel.from_pretrained, load_kwargs)
        )
        return model, unwrap_tokenizer(processor), FastLanguageModel.get_peft_model


def attach_lora(model, peft_fn, args):
    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": args.seed,
        "use_rslora": False,
        "loftq_config": None,
        "max_seq_length": args.max_seq_length,
        "target_modules": "all-linear",
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
    }
    return peft_fn(model, **_filter_kwargs(peft_fn, kwargs))


def build_sft_config(args, extra: dict[str, Any] | None = None):
    from trl import SFTConfig

    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": True,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "report_to": "none",
        "max_seq_length": args.max_seq_length,
        "dataset_num_proc": 1,
        "packing": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "max_grad_norm": 0.3,
    }
    if extra:
        kwargs.update(extra)
    if args.max_steps and args.max_steps > 0:
        kwargs["num_train_epochs"] = 1
    return SFTConfig(**_filter_kwargs(SFTConfig.__init__, kwargs))


def build_trainer(model, tokenizer, train_dataset, sft_args, data_collator=None):
    from trl import SFTTrainer

    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "args": sft_args,
    }
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if data_collator is not None and "data_collator" in params:
        kwargs["data_collator"] = data_collator
    return SFTTrainer(**kwargs)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.6-27B Unsloth 角色扮演 SFT")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="不建议。Qwen3.5/3.6 官方不推荐 QLoRA 4bit",
    )
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_chars", type=int, default=12000)
    parser.add_argument(
        "--loss",
        choices=["last_assistant", "all_assistant"],
        default="last_assistant",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def print_dry_run(tokenizer, row: dict, max_chars: int) -> None:
    messages = trim_messages(row["messages"], max_chars)
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    start = completion_char_start(text)
    prompt, completion = text[:start], text[start:]
    print("=== roles ===", [m["role"] for m in ensure_user_turn(messages)])
    print(f"=== prompt chars={len(prompt)} completion chars={len(completion)} ===")
    print("--- prompt tail ---")
    print(prompt[-400:].replace("\n", "\\n"))
    print("--- completion head ---")
    print(completion[:500].replace("\n", "\\n"))


def main(argv: Iterable[str] | None = None) -> None:
    _setup_cache_env()
    args = parse_args(argv)
    if not args.train_file.exists():
        raise FileNotFoundError(f"训练文件不存在: {args.train_file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        with args.train_file.open(encoding="utf-8") as f:
            row = json.loads(f.readline())
        print(f"dry_run model={args.model}")
        print_dry_run(tokenizer, row, args.max_chars)
        return

    from transformers import DataCollatorForSeq2Seq

    print(f"model={args.model}")
    print(f"train_file={args.train_file}")
    print(f"output_dir={args.output_dir}")
    print(
        f"loss={args.loss} max_seq_length={args.max_seq_length} "
        f"4bit={args.load_in_4bit} 16bit_lora={not args.load_in_4bit}"
    )

    model, tokenizer, peft_fn = load_qwen36(
        args.model,
        args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    model = attach_lora(model, peft_fn, args)

    raw = load_jsonl_dataset(args.train_file, args.max_samples)
    print(f"train samples: {len(raw)}")

    if args.loss == "last_assistant":
        tokenized = raw.map(
            lambda ex: tokenize_last_assistant(
                ex, tokenizer, args.max_seq_length, args.max_chars
            ),
            remove_columns=raw.column_names,
            desc="tokenize last-assistant",
        )
        tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 1)
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )
        sft_args = build_sft_config(args, extra={"max_seq_length": None})
        trainer = build_trainer(model, tokenizer, tokenized, sft_args, collator)
    else:
        text_ds = raw.map(
            lambda ex: tokenize_all_assistant_text(ex, tokenizer, args.max_chars),
            remove_columns=raw.column_names,
            desc="render chat template",
        )
        sft_args = build_sft_config(args, extra={"dataset_text_field": "text"})
        trainer = build_trainer(model, tokenizer, text_ds, sft_args)
        try:
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part=ASSISTANT_HEADER,
            )
        except Exception as exc:
            print(f"train_on_responses_only 不可用: {exc}")

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"saved LoRA adapter -> {args.output_dir}")

    if args.merge:
        merged = args.output_dir / "merged_16bit"
        print(f"merging 16bit -> {merged}")
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
