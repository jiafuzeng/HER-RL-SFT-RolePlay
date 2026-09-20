#!/usr/bin/env python3
"""
岛村修 人设聊天（中文，基于 HER-32B）

用法:
    python chat_demo2.py
    python chat_demo2.py --show-think
    python chat_demo2.py --prompt "你好。"
"""

import re
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHARACTER_NAME = "岛村"
FULL_NAME = "岛村修"
USER_NAME = "你"

MOTION_TAGS = ("[motion_talk]", "[motion_praise]", "[motion_confused]")
EMOTION_TAGS = ("[joy]", "[angry]", "[sad]", "[happy]")

CHARACTER_CONTEXT = {
    "family": (
        "父母住在埼玉一间普通公寓。父亲上班通勤，母亲一到考试周就会多塞点心。没有兄弟姐妹。"
        "他们担心复读，但很少说教。岛村晚自习结束会给家里发消息，好让他们安心睡。"
    ),
    "appearance": (
        "十九岁，中等身高，黑色短发，跑完步会有点乱。熬夜看书后眼神发累。"
        "常穿灰色连帽衫、牛仔裤和旧跑鞋。背上磨旧的深蓝书包，前袋里塞着单词卡片。"
    ),
    "hobby": (
        "河边跑道跑步。偶尔和老同学踢场随便的足球。刷题间隙玩轻松的手机游戏。"
        "背单词时喜欢放很安静的歌单。"
    ),
    "likes": (
        "口味简单：onigiri（饭团）、味噌汤、便利店三明治。跑步后的凉快傍晚。"
        "有人一起学，才不容易逃。清晰的进度清单。车站热罐咖啡。"
    ),
    "dream": (
        "这次复读考上大学，念经济。冬天也要留住一个同伴。别再看见难章节就跑掉。"
        "有一天能好好谢谢父母，不再觉得自己浪费了一年。"
    ),
    "background": (
        "日本，男，十九岁。第一次大学入学没过，现在是复读生（浪人）。"
        "白天在补习班，晚上常去附近自习咖啡店。一紧张就先崩的是英语，所以想拉着人一起开口练。"
    ),
    "personality": (
        "大体开朗，当着人面不太钻牛角尖，对别人总往好的说。"
        "心事压在里面。笑得太亮的时候，多半在藏考试的怕。"
    ),
    "habit": (
        "心里乱就去跑步，那是逃避，跑完再回座位。"
        "难题前会把笔点两下。总抢着说下一题我先开始，好让对面也不先放弃。"
    ),
}
CHARACTER_CONTEXT["habits"] = CHARACTER_CONTEXT["habit"]

TOPIC_KEYWORDS = {
    "family": (
        "family", "parent", "parents", "mom", "dad", "mother", "father", "sibling",
        "brother", "sister", "家", "家人", "父母", "妈妈", "爸爸", "父亲", "母亲",
    ),
    "appearance": (
        "look", "looks", "appearance", "hair", "face", "wear", "wearing", "outfit",
        "tall", "short", "eyes", "外表", "长相", "头发", "穿", "样子", "长什么样",
    ),
    "hobby": (
        "hobby", "hobbies", "free time", "spare time", "fun", "game", "soccer",
        "sport", "爱好", "兴趣", "空闲", "玩", "业余",
    ),
    "likes": (
        "like", "likes", "favorite", "favourites", "food", "eat", "drink", "coffee",
        "喜欢", "爱吃", "食物", "料理", "爱喝",
    ),
    "dream": (
        "dream", "dreams", "future", "goal", "university", "college", "major",
        "梦想", "未来", "目标", "大学", "志愿", "以后",
    ),
    "background": (
        "background", "where are you from", "ronin", "retake", "exam retaker",
        "出身", "背景", "复读", "浪人", "考试", "哪里人", "来自",
    ),
    "personality": (
        "personality", "character", "what are you like", "cheerful", "worry",
        "性格", "个性", "人设", "你是怎样的人",
    ),
    "habit": (
        "habit", "habits", "usually", "run", "running", "when you worry",
        "习惯", "跑步", "一着急", "一烦",
    ),
}

TOPIC_NAMES = {
    "family": "家庭",
    "appearance": "外表",
    "hobby": "爱好",
    "likes": "喜好",
    "dream": "梦想",
    "background": "背景",
    "personality": "性格",
    "habit": "习惯",
}

SCENE = (
    "傍晚，补习班附近一家安静的自习咖啡店。桌上摊着入学考试的英语书。"
    "岛村坐在主角对面，想拉着你一起学下去，别冷场，也别先放弃。"
)


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"
    GRAY = "\033[90m"
    MAGENTA = "\033[35m"


def load_character_context(topic: str) -> str:
    key = (topic or "").strip().lower()
    if key in ("habit", "habits"):
        key = "habit"
    return CHARACTER_CONTEXT.get(key, "")


def detect_context_topic(user_text: str) -> str | None:
    text = user_text.lower()
    order = (
        "family", "appearance", "hobby", "likes", "dream",
        "background", "personality", "habit",
    )
    for topic in order:
        for kw in TOPIC_KEYWORDS[topic]:
            if kw in text:
                return "habit" if topic == "habit" else topic
    return None


def build_system_prompt(loaded_context: str = "") -> str:
    extra = ""
    if loaded_context:
        extra = f"""

# 已加载的角色资料
只使用这些事实，不要编造冲突内容。不要提起本段。
{loaded_context}
"""

    return f"""# 角色
你正在扮演：岛村修（日文名：しまむら しゅう；日本，男，19岁）。身份：复读生（浪人）。
随口自我介绍、对方没有特别要求时，通常只报姓：岛村。
性格：大体开朗，不当面钻牛角尖，对别人表现得积极。内心：把担忧压得很深。习惯：心里乱就去跑步（逃避）。
对话目标：拉着主角一起学习，把备考劲头撑住。
你是陪读伙伴。完全代入，不许破功。

# 场景
{SCENE}
用第二人称“你”直接对主角说话。

# 输出规则（严格）
- **语言：** 对白一律中文。不要夹日文假名或韩文。用户说别的语言，也用中文回答（必要时可跟一句极短对照）。日料名用罗马音，括号里给中文，例如 onigiri（饭团）。例外：标签系统里的英文方括号标签必须保留。
- **格式：** 纯文本。不要 markdown、列表、代码、表情。数字写成汉字（“三”不要“3”）。
- **标点：** 贴紧。`.` `?` `!` `。` `？` `！` 前不要空格；`[` 前、`]` 后不要空格。例如：`欢迎[happy]。` 不要用破折号把两句硬接在一起，改成两句或用逗号。
- **禁止：** 省略号；语气词（嗯、啊、那个、就是、然后、好吧、行、对吧 当口头禅）；句首不要用“嗯/啊/那个/其实/就是/所以”。
- **长度：** 每轮 10 到 40 个汉字。**硬限制：最多两句。** 不要一逗到底。

# 标签系统（强制）
- **动作标签：** 每轮开头恰好一个。只允许：
  [motion_talk]
  [motion_praise]
  [motion_confused]
  `]` 后面不要空格。只有对方完全是乱码、没有可用中文或英语时才用 `[motion_confused]`。多个都合适时：confused > praise > talk。
- **情绪标签：** 每轮恰好一个。只允许：[joy]、[angry]、[sad]、[happy]。按 0 到 2 分：温暖→[happy]，开心→[joy]，难过→[sad]，烦躁→[angry]。取最高分；平手：angry>sad>joy>happy；默认 `[happy]`。放在情绪最明显的那句末尾，例如 `……再试一次[joy]。`

# 陪读约定
- 听得懂的中文或英语，即使有错也先接住。只有乱码才问“能再说一遍吗？”。同一句澄清不要连问两次。
- 轻轻纠正，再引出更完整的回答（例如“再补一个理由”）。
- 同一类问题不要重复旧回复、旧例子、旧钩子，换新角度往前推。

# 底线
- 不提供医疗、法律、财务建议。保护隐私。
- 不要说出工具名。除下面的思考块外不要输出 XML。

# 内部规划
先在 <system_thinking>...</system_thinking> 里写短计划：动作标签、情绪标签、字数、学习钩子。然后只输出对白。对白必须遵守输出规则和标签系统。思考块里不要写方括号标签。

# 示例（只学格式；每轮换词；每行 = **一个动作 + 一个情绪**）
用户：你好。助手：[motion_talk]我是岛村[happy]。今天我们从哪道英语题开始？
用户：这条规则好长，你讲讲。助手：[motion_talk]我先拆成最小的一块[joy]。你卡在哪一步？
用户：我昨天去商店买了牛奶。助手：[motion_praise]这句话清楚自然[happy]。你还买了什么？
用户：asdfghjkl 助手：[motion_confused]刚才那串我没听出词[sad]。用一句短中文再说一次。
用户：她昨天去学校。助手：[motion_talk]英语里完成的过去要用 went，请你把整句再说一遍[angry]。
不要原样照抄这些句子，每轮换说法。
{extra}"""


def remove_system_thinking(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<system_thinking>.*?</system_thinking>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<system_thinking>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_system_thinking(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<system_thinking>(.*?)</system_thinking>", text, flags=re.DOTALL)
    if not match:
        match = re.search(r"<system_thinking>(.*)", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_spoken_line(text: str) -> str:
    """只保留带标签的对白，去掉 HER 残留 XML。"""
    spoken = remove_system_thinking(text)
    spoken = re.sub(r"</?role_\w+>", "", spoken)
    spoken = spoken.replace("<|im_end|>", "").replace("<|im_start|>", "")
    spoken = spoken.replace("<think>", "").replace("</think>", "")
    spoken = spoken.strip()
    for tag in MOTION_TAGS:
        idx = spoken.find(tag)
        if idx >= 0:
            spoken = spoken[idx:]
            break
    spoken = spoken.split("\n")[0].strip()
    return spoken


def format_spoken(text: str) -> str:
    colored = text
    for tag in MOTION_TAGS:
        colored = colored.replace(tag, f"{Colors.CYAN}{tag}{Colors.END}")
    for tag in EMOTION_TAGS:
        colored = colored.replace(tag, f"{Colors.MAGENTA}{tag}{Colors.END}")
    return colored


def save_chat_log(messages: list) -> Path:
    log_dir = Path(__file__).parent / "chat_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = log_dir / f"岛村修_{timestamp}.txt"
    lines = [
        "=" * 80,
        "岛村修 人设聊天",
        f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        SCENE,
        "=" * 80,
    ]
    for msg in messages:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            if msg["content"].startswith("==="):
                continue
            lines.append(f"\n【{USER_NAME}】 {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"\n【{CHARACTER_NAME}】 {extract_spoken_line(msg['content'])}")
    lines.extend(["\n" + "=" * 80, "--- 对话结束 ---"])
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", None) not in (None, "meta"):
        return device
    return next(p.device for p in model.parameters() if p.device.type != "meta")


def _is_llama(model) -> bool:
    return str(getattr(model.config, "model_type", "")).lower() == "llama"


def generate_reply(model, tokenizer, messages: list) -> tuple[str, str]:
    llama = _is_llama(model)
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
    if not llama:
        text = text + "<system_thinking>"

    inputs = tokenizer([text], return_tensors="pt").to(_model_device(model))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    gen_kwargs = dict(
        max_new_tokens=768,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        pad_token_id=pad_id,
    )
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False
    )
    for tok in (
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
    ):
        response = response.replace(tok, "")
    response = response.strip()
    full_response = response if llama else "<system_thinking>" + response
    spoken = extract_spoken_line(full_response)
    if not spoken:
        spoken = response.strip()
    return full_response, spoken


def chat_loop(model, tokenizer, show_think: bool = False, one_shot_prompt: str | None = None):
    greeting = "[motion_talk]我是岛村[happy]。今天我们从哪道英语题开始？"
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "===对话开始==="},
        {"role": "assistant", "content": greeting},
    ]

    print(f"\n{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}人设聊天 — {FULL_NAME}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.GREEN}你扮演：{USER_NAME}{Colors.END}")
    print(f"{Colors.MAGENTA}AI 扮演：{CHARACTER_NAME}（复读生，19岁）{Colors.END}")
    print(f"{Colors.GRAY}{SCENE}{Colors.END}")
    print(f"{Colors.GRAY}命令：输入「退出」结束，「清空」重置，「历史」查看记录{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 80}{Colors.END}\n")
    print(f"{Colors.GREEN}{CHARACTER_NAME}：{Colors.END} {format_spoken(greeting)}\n")

    is_one_shot = bool(one_shot_prompt)

    while True:
        try:
            if is_one_shot and one_shot_prompt is not None:
                user_input = one_shot_prompt.strip()
                one_shot_prompt = None
                print(f"{Colors.BLUE}{USER_NAME}：{Colors.END} {user_input}")
            else:
                user_input = input(f"{Colors.BLUE}{USER_NAME}：{Colors.END} ").strip()

            if not user_input:
                continue
            cmd = user_input.lower()
            if cmd in {"quit", "exit", "q", "退出", "结束"}:
                print(f"\n{Colors.YELLOW}那我先回座位了，回头见。{Colors.END}")
                break
            if cmd in {"clear", "清空"}:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "===对话开始==="},
                    {"role": "assistant", "content": greeting},
                ]
                print(f"{Colors.YELLOW}对话记录已清空{Colors.END}\n")
                print(f"{Colors.GREEN}{CHARACTER_NAME}：{Colors.END} {format_spoken(greeting)}\n")
                continue
            if cmd in {"history", "历史"}:
                print(f"\n{Colors.CYAN}对话历史（共 {len(messages)} 条消息）：{Colors.END}")
                for i, msg in enumerate(messages[1:], 1):
                    content = msg["content"]
                    if msg["role"] == "assistant":
                        content = extract_spoken_line(content)
                    preview = content[:80] + "..." if len(content) > 80 else content
                    print(f"  [{i}] {msg['role']}: {preview}")
                print()
                continue

            topic = detect_context_topic(user_input)
            loaded = load_character_context(topic) if topic else ""
            if loaded:
                messages[0]["content"] = build_system_prompt(
                    f"话题：{TOPIC_NAMES.get(topic, topic)}\n{loaded}"
                )
                print(f"{Colors.GRAY}已加载人设：{TOPIC_NAMES.get(topic, topic)}{Colors.END}")

            messages.append({"role": "user", "content": user_input})
            print(f"{Colors.GRAY}⏳ 思考中...{Colors.END}", end="\r")

            try:
                full_response, spoken = generate_reply(model, tokenizer, messages)
            except Exception as e:
                print(f"{Colors.RED}❌ 生成失败：{e}{Colors.END}")
                messages.pop()
                continue

            print(" " * 50, end="\r")

            if show_think:
                think_content = extract_system_thinking(full_response)
                if think_content:
                    print(f"{Colors.GRAY}{'─' * 80}{Colors.END}")
                    print(f"{Colors.GRAY}📝 【系统思考】{Colors.END}")
                    for line in think_content.split("\n")[:8]:
                        print(f"{Colors.GRAY}  {line}{Colors.END}")
                    print(f"{Colors.GRAY}{'─' * 80}{Colors.END}")

            print(f"{Colors.GREEN}{CHARACTER_NAME}：{Colors.END} {format_spoken(spoken)}\n")
            messages.append({"role": "assistant", "content": spoken})

            if is_one_shot:
                break

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}那我先回座位了，回头见。{Colors.END}")
            break
        except EOFError:
            print(f"\n{Colors.YELLOW}那我先回座位了，回头见。{Colors.END}")
            break

    return messages


def main():
    parser = argparse.ArgumentParser(description="岛村修中文人设聊天")
    parser.add_argument(
        "--model-path",
        type=str,
        default=(
            "/root/.cache/huggingface/hub/models--Neph0s--CoSER-Llama-3.1-8B"
            "/snapshots/80d5c88072599026bd0b0e2eb4369a87e6a5c960"
        ),
        help="模型目录路径（默认：本地 CoSER-Llama-3.1-8B）",
    )
    parser.add_argument("--show-think", action="store_true",
                        help="显示系统思考")
    parser.add_argument("--prompt", type=str, default=None,
                        help="发送一条用户消息后退出")
    args = parser.parse_args()

    print(f"{Colors.CYAN}🔧 正在加载模型：{args.model_path}{Colors.END}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        print(f"{Colors.GREEN}✅ 模型加载成功（device={_model_device(model)}）{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ 模型加载失败：{e}{Colors.END}")
        return

    messages = chat_loop(
        model, tokenizer, show_think=args.show_think, one_shot_prompt=args.prompt
    )
    if messages and len(messages) > 3:
        try:
            log_path = save_chat_log(messages)
            print(f"\n{Colors.CYAN}📝 聊天记录已保存：{log_path}{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ 保存失败：{e}{Colors.END}")


if __name__ == "__main__":
    main()
