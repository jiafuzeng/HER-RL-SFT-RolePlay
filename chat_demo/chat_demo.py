#!/usr/bin/env python3
"""
HER 模型交互式角色扮演聊天演示
与经典文学作品中的角色进行情景对话。

用法:
    python chat_demo.py
    python chat_demo.py --show-think
    python chat_demo.py --show-rolethink
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 终端输出颜色
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'
    MAGENTA = '\033[35m'


def remove_system_thinking(text: str) -> str:
    """移除 <system_thinking>...</system_thinking> 标签及其内容"""
    if not text:
        return text
    pattern = r'<system_thinking>.*?</system_thinking>\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<system_thinking>.*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_system_thinking(text: str) -> str:
    """提取 system_thinking 内容（去掉其中可能泄漏的角色标签）"""
    if not text:
        return ""
    match = re.search(r'<system_thinking>(.*?)</system_thinking>', text, flags=re.DOTALL)
    if not match:
        match = re.search(r'<system_thinking>(.*)', text, flags=re.DOTALL)
    if match:
        content = match.group(1).strip()
        # 去掉可能泄漏进去的角色标签
        content = re.sub(r'</?role_\w+>', '', content)
        return content
    return ""


def format_for_display(text: str, show_rolethink: bool = True) -> str:
    """格式化展示：role_thinking 用 []，role_action 用 ()"""
    if not text:
        return text

    result = text

    # 处理 role_thinking
    if show_rolethink:
        result = result.replace('<role_thinking>', '[').replace('</role_thinking>', ']')
    else:
        result = re.sub(r'<role_thinking>.*?</role_thinking>', '', result, flags=re.DOTALL)

    # 将 role_action 替换为 ()
    result = result.replace('<role_action>', '(').replace('</role_action>', ')')
    result = result.replace('<role_speech>', '').replace('</role_speech>', '')

    return result.strip()


def load_sample_scenarios(use_coser: bool = True):
    """加载或创建示例场景

    Args:
        use_coser: 为 True 时优先加载 CoSER 场景（200 个经典文学场景）
    """
    # 优先尝试加载 CoSER 场景
    if use_coser:
        coser_file = Path(__file__).parent / "coser_scenarios.json"
        if coser_file.exists():
            print(f"{Colors.CYAN}📚 正在加载 CoSER 场景（200 个经典文学场景）...{Colors.END}")
            with open(coser_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    # 否则使用内置场景
    scenarios_file = Path(__file__).parent / "scenarios.json"

    # 若场景文件已存在则直接加载
    if scenarios_file.exists():
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 否则创建示例场景
    scenarios = [
        {
            "book": "傲慢与偏见",
            "topic": "班纳特先生就达西先生的求婚质问伊丽莎白",
            "scenario": "场景设在班纳特先生的私人书房，这里满是皮面精装书，是他安静沉思的避风港。伊丽莎白被突然召来，班纳特先生手中握着一封信，那神情带着他一贯的冷嘲热讽。",
            "character_profiles": {
                "班纳特先生": "伊丽莎白的父亲，以刻薄的机智和超然态度著称。才智过人、博览群书，更爱躲进书房独处。说话尖刻，惯用冷幽默。",
                "伊丽莎白·班纳特": "故事主角，聪慧而意志坚定。反应敏捷，带着戏谑的幽默感。珍视诚实与正直，压力之下仍能保持从容。"
            },
            "key_characters": [
                {
                    "name": "班纳特先生",
                    "thought": "达西这件事很微妙。我得探清伊丽莎白的真实心意，又不能显得过于多愁善感。"
                },
                {
                    "name": "伊丽莎白·班纳特",
                    "thought": "父亲这个时候叫我过来很不寻常。但愿不是为了凯瑟琳夫人来访的事。"
                }
            ]
        },
        {
            "book": "了不起的盖茨比",
            "topic": "尼克·卡拉威在一场奢华派对上遇见盖茨比",
            "scenario": "盖茨比府邸的派对正热闹进行。爵士乐弥漫空中，香槟不断流淌，衣着考究的宾客在草坪上寒暄。尼克独自徘徊，旁观这场盛宴，却在图书室旁遇见一个神秘的男人。",
            "character_profiles": {
                "杰伊·盖茨比": "神秘的百万富翁，热衷举办奢华派对。优雅表象之下是执着于重温往昔的浪漫梦想家。迷人，却极度孤独。",
                "尼克·卡拉威": "故事叙述者，来自中西部的耶鲁毕业生。诚实、宽容，倾向于先不作评判。既被周围的奢靡吸引，又对其感到厌恶。"
            },
            "key_characters": [
                {
                    "name": "杰伊·盖茨比",
                    "thought": "又一场派对，又一个等待的夜晚。也许今晚她会来。我必须维持体面。"
                },
                {
                    "name": "尼克·卡拉威",
                    "thought": "我还从没见过东道主。这些派对华丽非凡，可这一切狂欢里总有种空洞。"
                }
            ]
        }
    ]

    # 保存场景以便后续使用
    with open(scenarios_file, 'w', encoding='utf-8') as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    return scenarios


def print_scenarios(scenarios: list):
    """打印可选场景"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}📚 可选场景{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    for i, s in enumerate(scenarios):
        print(f"{Colors.CYAN}[{i}]{Colors.END} {Colors.BOLD}📖 {s['book']}{Colors.END}")
        print(f"    {Colors.GRAY}{s['topic']}{Colors.END}")
        chars = list(s['character_profiles'].keys())
        print(f"    {Colors.MAGENTA}👥 角色：{', '.join(chars)}{Colors.END}")
        print()


def print_characters(scenario: dict):
    """打印该场景中的可选角色"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}👥 可选角色 - {scenario['book']}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    for i, (name, profile) in enumerate(scenario['character_profiles'].items()):
        print(f"{Colors.CYAN}[{i}]{Colors.END} {Colors.BOLD}{name}{Colors.END}")
        preview = profile[:150] + "..." if len(profile) > 150 else profile
        print(f"    {Colors.GRAY}{preview}{Colors.END}")
        print()


def build_system_prompt(scenario: dict, character_name: str, user_character_name: str) -> str:
    """构建角色系统提示词"""
    book = scenario['book']
    scene = scenario['scenario']
    profiles = scenario['character_profiles']

    char_profile = profiles.get(character_name, "")
    user_profile = profiles.get(user_character_name, "一个正在与该角色互动的人。")

    # 查找角色的初始想法
    char_thought = ""
    for kc in scenario['key_characters']:
        if kc['name'] == character_name:
            char_thought = kc.get('thought', '')
            break

    prompt = f"""你正在扮演来自《{book}》的{character_name}。

==={character_name}的人物设定===
{char_profile}

===当前场景===
{scene}

===你此刻的想法===
{char_thought}

===你正在互动的对象===
{user_character_name}：{user_profile}

===要求===
- 始终保持{character_name}的身份与口吻
- 回复自然、引人入胜，风格与原著一致
- 从{character_name}的视角作出回应
- **重要：请用第二人称“你”直接对“{user_character_name}”说话，不要用第三人称转述。**
- **请用中文进行角色扮演：内心想法、动作描写和对白均使用中文。**

===输出格式===
你的输出应包含思考、对白和动作，采用如下两部分结构：

1. 系统思考：最开头的一个独立块，用 <system_thinking> 和 </system_thinking> 包裹。这是第三人称分析，用于规划如何扮演该角色。

2. 角色扮演回复：角色的实际回应，包括：
   - <role_thinking>内心想法</role_thinking>（他人不可见）
   - <role_action>肢体动作</role_action>（他人可见）
   - 对白（纯文本，角色说出口的话）"""

    return prompt


def save_chat_log(messages: list, scenario: dict, character_name: str, user_character: str):
    """将聊天记录保存到文件"""
    log_dir = Path(__file__).parent / "chat_logs"
    log_dir.mkdir(exist_ok=True)

    safe_book = re.sub(r'[^\w\-]', '_', scenario['book'])[:30]
    safe_char = re.sub(r'[^\w\-]', '_', character_name)[:20]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_book}_{safe_char}_{timestamp}.txt"
    filepath = log_dir / filename

    lines = [
        "=" * 80,
        "HER 聊天演示 - 对话记录",
        "=" * 80,
        f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"作品：{scenario['book']}",
        f"AI 扮演：{character_name}",
        f"用户扮演：{user_character}",
        "=" * 80,
        "",
        "【场景】",
        scenario['scenario'][:500],
        "",
        "=" * 80,
        "【对话】",
        "=" * 80,
    ]

    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            continue
        elif role == 'user':
            if "===Conversation Start===" not in content and "===对话开始===" not in content:
                lines.append(f"\n【{user_character}】")
                lines.append(content)
        elif role == 'assistant':
            lines.append(f"\n【{character_name}】")
            lines.append(content)

    lines.extend(["\n" + "=" * 80, "--- 对话结束 ---"])

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return filepath


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", None) not in (None, "meta"):
        return device
    return next(p.device for p in model.parameters() if p.device.type != "meta")


def generate_reply(model, tokenizer, messages: list) -> tuple[str, str]:
    """生成一条 HER 格式回复，返回 (full_response, clean_response)。"""
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
    text = text + "<system_thinking>"

    inputs = tokenizer([text], return_tensors="pt").to(_model_device(model))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pad_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    full_response = "<system_thinking>" + response
    return full_response, remove_system_thinking(full_response)


def chat_loop(model, tokenizer, scenario: dict, character_name: str, user_character: str,
              show_think: bool = False, show_rolethink: bool = True,
              one_shot_prompt: str | None = None):
    """主聊天循环"""
    book = scenario['book']

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}🎭 开始对话 - {book}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}你扮演：{user_character}{Colors.END}")
    print(f"{Colors.MAGENTA}AI 扮演：{character_name}{Colors.END}")
    print(f"{Colors.GRAY}显示系统思考：{'是' if show_think else '否'}{Colors.END}")
    print(f"{Colors.GRAY}显示角色内心：{'是' if show_rolethink else '否'}{Colors.END}")
    print(f"{Colors.GRAY}命令：输入「退出」结束，「清空」重置，「历史」查看记录{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    # 展示场景
    print(f"{Colors.CYAN}📍 场景：{Colors.END}")
    print(f"{Colors.GRAY}{scenario['scenario'][:300]}...{Colors.END}\n")

    # 构建消息
    system_prompt = build_system_prompt(scenario, character_name, user_character)

    # 开场问候
    greeting = f"*{character_name}望向你*"
    for kc in scenario.get('key_characters', []):
        if kc['name'] == character_name:
            greeting = f"*走进场景* 你好，{user_character}。"
            break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "===对话开始==="},
        {"role": "assistant", "content": greeting}
    ]

    print(f"{Colors.GREEN}{character_name}：{Colors.END} {greeting}\n")

    is_one_shot = bool(one_shot_prompt)

    while True:
        try:
            if is_one_shot and one_shot_prompt is not None:
                user_input = one_shot_prompt.strip()
                one_shot_prompt = None
                print(f"{Colors.BLUE}{user_character}：{Colors.END} {user_input}")
            else:
                user_input = input(f"{Colors.BLUE}{user_character}：{Colors.END} ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q', '退出', '结束']:
                print(f"\n{Colors.YELLOW}👋 再见！{Colors.END}")
                break

            if user_input.lower() in ['clear', '清空']:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "===对话开始==="},
                    {"role": "assistant", "content": greeting}
                ]
                print(f"{Colors.YELLOW}🔄 对话记录已清空{Colors.END}\n")
                print(f"{Colors.GREEN}{character_name}：{Colors.END} {greeting}\n")
                continue

            if user_input.lower() in ['history', '历史']:
                print(f"\n{Colors.CYAN}📜 对话历史（共 {len(messages)} 条消息）：{Colors.END}")
                for i, msg in enumerate(messages[1:], 1):
                    content = msg['content'][:80] + '...' if len(msg['content']) > 80 else msg['content']
                    print(f"  [{i}] {msg['role']}: {content}")
                print()
                continue

            # 加入用户消息
            messages.append({"role": "user", "content": user_input})

            # 生成回复
            print(f"{Colors.GRAY}⏳ 思考中...{Colors.END}", end='\r')

            try:
                full_response, clean_response = generate_reply(model, tokenizer, messages)
            except Exception as e:
                print(f"{Colors.RED}❌ 生成失败：{e}{Colors.END}")
                messages.pop()
                continue

            print(" " * 50, end='\r')

            # 按需展示系统思考
            if show_think:
                think_content = extract_system_thinking(full_response)
                if think_content:
                    print(f"\n{Colors.GRAY}{'─'*80}{Colors.END}")
                    print(f"{Colors.GRAY}📝 【系统思考】{Colors.END}")
                    print(f"{Colors.GRAY}{'─'*80}{Colors.END}")
                    for line in think_content.split('\n')[:10]:  # 限制行数
                        print(f"{Colors.GRAY}  {line}{Colors.END}")
                    print(f"{Colors.GRAY}{'─'*80}{Colors.END}\n")

            # 展示角色回复
            print(f"{Colors.GREEN}{'═'*80}{Colors.END}")
            print(f"{Colors.GREEN}🎭 【{character_name}的回复】{Colors.END}")
            print(f"{Colors.GREEN}{'═'*80}{Colors.END}")
            display_response = format_for_display(clean_response, show_rolethink=show_rolethink)
            print(f"{Colors.GREEN}{display_response}{Colors.END}")
            print(f"{Colors.GREEN}{'═'*80}{Colors.END}\n")

            messages.append({"role": "assistant", "content": clean_response})

            if is_one_shot:
                break

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}👋 再见！{Colors.END}")
            break
        except EOFError:
            print(f"\n{Colors.YELLOW}👋 再见！{Colors.END}")
            break

    return messages


def main():
    parser = argparse.ArgumentParser(description="HER 模型交互式角色扮演聊天演示")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/HER-32B",
                        help="模型目录路径（默认：/root/autodl-tmp/HER-32B）")
    parser.add_argument("--show-think", action="store_true",
                        help="显示系统思考（system_thinking）")
    parser.add_argument("--show-rolethink", action="store_true",
                        help="显示角色内心（role_thinking，默认隐藏）")
    parser.add_argument("--scenario", type=int, default=None,
                        help="场景编号（默认：交互式选择）")
    parser.add_argument("--character", type=int, default=None,
                        help="角色编号（默认：交互式选择）")
    parser.add_argument("--simple", action="store_true",
                        help="使用内置简易场景，而非 CoSER 数据集")
    parser.add_argument("--user-character", type=int, default=None,
                        help="用户扮演角色编号（默认：交互式选择）")
    parser.add_argument("--prompt", type=str, default=None,
                        help="发送一条用户消息后退出（用于冒烟测试）")

    args = parser.parse_args()

    # 加载场景
    scenarios = load_sample_scenarios(use_coser=not args.simple)
    print(f"{Colors.GREEN}✅ 已加载 {len(scenarios)} 个场景{Colors.END}")

    # 选择场景
    if args.scenario is not None:
        if 0 <= args.scenario < len(scenarios):
            scenario = scenarios[args.scenario]
        else:
            print(f"{Colors.RED}❌ 无效的场景编号{Colors.END}")
            return
    else:
        print_scenarios(scenarios)
        while True:
            try:
                idx = int(input(f"{Colors.CYAN}请选择场景（0-{len(scenarios)-1}）：{Colors.END}"))
                if 0 <= idx < len(scenarios):
                    scenario = scenarios[idx]
                    break
                print(f"{Colors.RED}编号无效{Colors.END}")
            except (ValueError, KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}👋 再见！{Colors.END}")
                return

    print(f"\n{Colors.GREEN}✅ 已选择：{scenario['book']}{Colors.END}")

    # 选择角色
    char_names = list(scenario['character_profiles'].keys())
    print_characters(scenario)

    if args.character is not None:
        if 0 <= args.character < len(char_names):
            character_name = char_names[args.character]
        else:
            print(f"{Colors.RED}❌ 无效的角色编号{Colors.END}")
            return
    else:
        while True:
            try:
                idx = int(input(f"{Colors.CYAN}请选择 AI 扮演的角色（0-{len(char_names)-1}）：{Colors.END}"))
                if 0 <= idx < len(char_names):
                    character_name = char_names[idx]
                    break
                print(f"{Colors.RED}编号无效{Colors.END}")
            except (ValueError, KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}👋 再见！{Colors.END}")
                return

    # 选择用户扮演的角色
    remaining_chars = [c for c in char_names if c != character_name]
    if remaining_chars:
        if args.user_character is not None:
            if 0 <= args.user_character < len(remaining_chars):
                user_character = remaining_chars[args.user_character]
            else:
                print(f"{Colors.RED}❌ 无效的用户角色编号{Colors.END}")
                return
        else:
            print(f"\n{Colors.CYAN}你想扮演谁？{Colors.END}")
            for i, c in enumerate(remaining_chars):
                print(f"  [{i}] {c}")
            print(f"  [{len(remaining_chars)}] 自定义名字")

            while True:
                try:
                    idx = int(input(f"{Colors.CYAN}请选择（0-{len(remaining_chars)}）：{Colors.END}"))
                    if idx == len(remaining_chars):
                        user_character = input(f"{Colors.CYAN}你的名字：{Colors.END}").strip() or "用户"
                        break
                    elif 0 <= idx < len(remaining_chars):
                        user_character = remaining_chars[idx]
                        break
                except (ValueError, KeyboardInterrupt, EOFError):
                    print(f"\n{Colors.YELLOW}👋 再见！{Colors.END}")
                    return
    else:
        user_character = "用户"

    print(f"\n{Colors.GREEN}✅ AI 扮演：{character_name}{Colors.END}")
    print(f"{Colors.BLUE}✅ 你扮演：{user_character}{Colors.END}")

    # 加载模型
    print(f"\n{Colors.CYAN}🔧 正在加载模型：{args.model_path}{Colors.END}")
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

    # 开始聊天
    messages = chat_loop(
        model,
        tokenizer,
        scenario,
        character_name,
        user_character,
        show_think=args.show_think,
        show_rolethink=args.show_rolethink,
        one_shot_prompt=args.prompt,
    )

    # 保存记录
    if messages and len(messages) > 3:
        try:
            log_path = save_chat_log(messages, scenario, character_name, user_character)
            print(f"\n{Colors.CYAN}📝 聊天记录已保存：{log_path}{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ 保存失败：{e}{Colors.END}")


if __name__ == "__main__":
    main()
