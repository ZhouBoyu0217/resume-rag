"""
RAG 检索引擎：把「检索 + 生成」封装成可被调用的函数。
- 命令行直接跑本文件：交互式问答
- 被 main.py（FastAPI 接口）import：把 answer() 挂到接口上
"""
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# 复用 rag_core 的向量化能力，避免重复实现
from rag_core import embed, collection

load_dotenv()

# 生成用的客户端（embedding 客户端在 rag_core 里）
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def retrieve(question, top_k=6, resume_id=None):
    """检索：问题转向量 → 找最相似的 top_k 块。

    resume_id 不为空时加元数据过滤（where），只在那一份简历的块里找。
    这就是"元数据过滤"——不用单独建库，靠 Chroma 的 where 条件就能做到。
    """
    q_vec = embed([question])[0]          # 复用 rag_core.embed（批量接口，传单元素列表）
    result = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where={"resume_id": resume_id} if resume_id else None,
        include=["documents", "distances", "metadatas"],
    )
    result["metadatas"] = result.get("metadatas") or [[{}] * len(result["documents"][0])]
    return result["documents"][0], result["distances"][0], result["metadatas"][0]


SYSTEM_PROMPT = (
    "你是简历助手。只根据下面提供的简历内容回答问题，"
    "回答要具体（引用简历里的项目和数字）；"
    "如果简历里确实没有相关信息，就直说'简历里没有提到'，不要编造。\n\n"
    "排版要求（前端会按 Markdown 渲染）：\n"
    "- 用 **加粗** 标出关键结论、数字、指标、技术名词\n"
    "- 用 `反引号` 标出具体的技术栈/工具名\n"
    "- 有条目时用 - 或 1. 列成列表，不要挤成一段\n"
    "- 不要输出标题层级（#），不要用表格\n"
)


def _to_sse(event, data):
    """把一条事件打包成 SSE 格式：event: xxx\\ndata: {...}\\n\\n"""
    import json as _json
    return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"


def stream_answer(question, top_k=6, resume_id=None):
    """流式版检索+生成：边生成边吐字，通过 SSE 推给前端。

    事件顺序：
      event: sources  → 先告诉前端"检索到了哪些块"（列表，含 distance / 简历名）
      event: delta    → 重复多次，每次一小段文字（拼起来就是完整答案）
      event: done     → 结束信号
      event: error    → 出错时的错误信息

    为什么先给 sources 再给正文？
    检索（retrieve）在生成（generate）之前就完成了，前端可以先把引用卡片画出来，
    不必等整个答案结束。
    """
    # 第 1 步：检索（同步、快，通常 0.3-1s）
    context, dists, metas = retrieve(question, top_k=top_k, resume_id=resume_id)
    if not context:
        yield _to_sse("delta", {"text": "库里没有可检索的简历内容，先上传一份吧。"})
        yield _to_sse("done", {"ok": True})
        return

    sources = []
    for doc, d, m in zip(context, dists, metas):
        sources.append({
            "text": doc,
            "distance": round(float(d), 4),
            "resume_name": (m or {}).get("resume_name", ""),
        })
    yield _to_sse("sources", sources)

    # 第 2 步：流式生成
    context_text = "\n\n".join(context)
    scope = ""
    if resume_id and metas and metas[0].get("resume_name"):
        scope = f"下面是候选人「{metas[0]['resume_name']}」的简历内容。\n"
    try:
        stream = chat_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"{scope}简历内容：\n{context_text}\n\n问题：{question}",
                },
            ],
            stream=True,          # 关键：流式模式
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield _to_sse("delta", {"text": delta})
    except Exception as e:
        yield _to_sse("error", {"message": str(e)})
        return

    yield _to_sse("done", {"ok": True})


def answer(question, top_k=6, with_sources=False, resume_id=None):
    """检索 + 生成。

    with_sources=False（默认）→ 直接返回回答文本（命令行用）
    with_sources=True          → 返回 {"answer": ..., "sources": [...]}（接口用，带引用溯源）
    resume_id                  → 只检索这一份简历；None = 检索全部
    """
    context, dists, metas = retrieve(question, top_k=top_k, resume_id=resume_id)
    context_text = "\n\n".join(context)

    scope = ""
    if resume_id and metas and metas[0].get("resume_name"):
        scope = f"下面是候选人「{metas[0]['resume_name']}」的简历内容。\n"

    resp = chat_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"{scope}简历内容：\n{context_text}\n\n问题：{question}",
            },
        ],
    )
    text = resp.choices[0].message.content

    if not with_sources:
        return text

    # 引用溯源：把这次用到的原文块一并返回，方便前端展示"依据是什么"
    sources = [
        {"text": doc, "distance": round(float(d), 4),
         "resume_name": (m or {}).get("resume_name", "")}
        for doc, d, m in zip(context, dists, metas)
    ]
    return {"answer": text, "sources": sources}


if __name__ == "__main__":
    print("输入问题开始问答；输入 d 切换'显示引用来源'；输入 exit 退出。")
    show = False
    while True:
        q = input("\n问：")
        if q.strip() in ("exit", "quit", "q"):
            break
        if q.strip() == "d":
            show = not show
            print(f"[引用来源已{'开启' if show else '关闭'}]")
            continue

        if show:
            result = answer(q, with_sources=True)
            print("答：", result["answer"])
            print("\n--- 依据的简历片段 ---")
            for i, s in enumerate(result["sources"], 1):
                preview = s["text"][:50].replace("\n", " / ")
                print(f"  [{i}] 距离={s['distance']}  {preview}")
        else:
            print("答：", answer(q))
