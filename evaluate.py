"""
RAG 评测脚本：用固定测试集 + LLM 当裁判，量化系统的回答质量。

为什么要评测？
- 没有评测，"效果变好还是变差"全凭感觉
- 每次改了切分/检索/提示词，跑一遍这个脚本，分数变化一目了然

用法：python evaluate.py
输出：每条测试题的得分 + 总体平均分，并写入 eval_report.json
"""
import json
from query import answer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
judge = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 测试集：每题标注"期望要点"（回答里应出现的关键信息）
TEST_CASES = [
    {
        "question": "我有哪些项目经历？",
        "expect": ["企业级智能体", "Agent平台", "法律合同", "审查助手"],
    },
    {
        "question": "我熟悉哪些技术？",
        "expect": ["LangChain", "RAG", "FastAPI", "Python"],
    },
    {
        "question": "我什么时候毕业的？",
        "expect": ["2023"],
    },
    {
        "question": "他做过的项目里，哪个和法务有关？",
        "expect": ["法律合同", "审查"],
    },
    {
        "question": "他的 GPA 是多少？",
        "expect": ["没有提到", "未提及", "无法", "不知道"],   # 应老实说没有
    },
    {
        "question": "他在腾讯工作过吗？",
        "expect": ["没有", "未", "不是"],
    },
]


def judge_answer(question, expected, got):
    """用大模型当裁判：判断回答是否覆盖了期望要点、是否编造。返回 0-10 分。"""
    prompt = f"""你是严格的评测员。根据下面的信息给一个回答打分（0-10 分）。

问题：{question}
期望回答包含的要点：{expected}
系统的回答：{got}

评分标准：
- 覆盖了所有要点、且没有编造 → 9-10 分
- 覆盖大部分要点 → 6-8 分
- 覆盖少部分 / 有轻微编造 → 3-5 分
- 完全错误或严重编造 → 0-2 分

只输出一个 JSON：{{"score": 分数, "reason": "一句话理由"}}"""

    resp = judge.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()

    # 容错解析（model 可能包 ```json）
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    try:
        data = json.loads(raw)
        return int(data.get("score", 0)), data.get("reason", "")
    except Exception:
        return 0, f"解析失败：{raw[:80]}"


def main():
    results = []
    print("=" * 60)
    print("RAG 评测开始\n")

    for i, case in enumerate(TEST_CASES, 1):
        q = case["question"]
        print(f"[{i}/{len(TEST_CASES)}] {q}")

        result = answer(q, with_sources=True)
        text = result["answer"]

        score, reason = judge_answer(q, case["expect"], text)
        print(f"    得分：{score}/10 — {reason}")

        results.append({
            "question": q,
            "answer": text,
            "expected": case["expect"],
            "score": score,
            "reason": reason,
        })

    avg = sum(r["score"] for r in results) / len(results)
    print("\n" + "=" * 60)
    print(f"平均分：{avg:.1f}/10")
    print("=" * 60)

    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"avg": avg, "results": results}, f, ensure_ascii=False, indent=2)
    print("详细报告已写入 eval_report.json")


if __name__ == "__main__":
    main()
