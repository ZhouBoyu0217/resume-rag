"""
摘要层：解决 RAG 处理不了"全局问题"的局限。

为什么需要它？
- RAG 靠 top_k 检索，只能捞到几块，覆盖不了所有简历
- 但"所有简历的共同点""谁的经验最匹配某岗位"这类问题，需要看全部内容
- 解法（Map-Reduce）：每份简历离线生成一份摘要 → 提问时把所有摘要一起喂给大模型

数据存放：摘要存在 storage/summaries/{resume_id}.txt，与原文同目录体系
"""
import os
import glob
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv
import rag_core

load_dotenv()

chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SUMMARY_DIR = os.path.join(rag_core.STORAGE_DIR, "summaries")
os.makedirs(SUMMARY_DIR, exist_ok=True)


def make_summary(text):
    """Map 阶段：用大模型把一份简历压成 200 字左右的结构化摘要。"""
    resp = chat_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是简历分析助手。把这份简历压缩成 150-200 字的结构化摘要，"
                    "必须包含：姓名（如简历中有）、工作年限、核心技术栈、"
                    "主要项目（含量化指标）、求职方向。只输出摘要正文，不要寒暄。"
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip()


def ensure_summary(resume_id, resume_name):
    """保证这份简历有摘要；没有就生成一份并落盘。返回摘要文本。"""
    path = os.path.join(SUMMARY_DIR, f"{resume_id}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    raw = rag_core.get_raw(resume_id)
    if raw is None:
        return f"（简历「{resume_name}」原文缺失，无法生成摘要）"

    summary = make_summary(raw)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    return summary


def get_all_summaries():
    """取库里所有简历的摘要（没有的现场补生成）。返回 [{"name","summary"}...]"""
    resumes = rag_core.list_resumes()
    result = []
    for r in resumes:
        summary = ensure_summary(r["resume_id"], r["resume_name"])
        result.append({"resume_name": r["resume_name"], "summary": summary})
    return result


def compare(question):
    """Reduce 阶段：把所有摘要 + 用户问题一起交给大模型，做全局分析。"""
    summaries = get_all_summaries()
    if not summaries:
        return {"answer": "库里还没有简历，先上传几份吧。", "count": 0}

    corpus = "\n\n".join(
        f"【{s['resume_name']}】\n{s['summary']}" for s in summaries
    )

    resp = chat_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是候选人分析助手。根据提供的多份简历摘要回答问题。"
                    "回答要结构化：先给结论，再分点说明，并点名提到具体是哪位候选人。"
                    "如果摘要信息不足以回答，就明确说明缺什么信息。\n"
                    "排版：用 **加粗** 标出关键结论和数字，条目用 - 或 1. 列成列表，"
                    "不要输出 # 标题，不要用表格。"
                ),
            },
            {
                "role": "user",
                "content": f"候选人简历摘要：\n{corpus}\n\n问题：{question}",
            },
        ],
    )
    return {"answer": resp.choices[0].message.content, "count": len(summaries)}
