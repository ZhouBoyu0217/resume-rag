"""
RAG 简历问答服务（FastAPI 接口层）

启动：uvicorn main:app --reload --port 8001   （8000 被 open-ai 网关占着）
访问：http://127.0.0.1:8001/docs

设计要点：
- resume_id 是唯一身份（时间戳+文件名），resume_name 只做展示
- 上传时"双写"：原文存 storage/，切分块存 Chroma
- 同名简历拒绝上传（避免误覆盖同名不同人）
"""
import time
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query import answer, collection, stream_answer
import rag_core
import summary_core
import analyzer


app = FastAPI(title="简历问答服务", description="基于 RAG 的简历知识库问答")


# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    question: str
    top_k: int = 6
    resume_id: str | None = None   # 只查某一份简历；None = 查全部


# ---------- 接口 1：问答 ----------
@app.post("/chat")
def chat(req: ChatRequest):
    """收一个问题，返回 RAG 生成的回答 + 引用来源。
    传入 resume_id 时只看这一份简历；不传则检索全部。"""
    return answer(req.question, top_k=req.top_k, with_sources=True,
                  resume_id=req.resume_id)


# ---------- 接口 1b：流式问答（SSE）----------
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """流式版 /chat：用 Server-Sent Events 边生成边推字。

    前端用 fetch + ReadableStream 一块块读，
    先收到 sources 事件（引用卡片立即可显示），再收到一串 delta（正文逐字浮现）。
    """
    return StreamingResponse(
        stream_answer(req.question, top_k=req.top_k, resume_id=req.resume_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 防止 nginx 等代理缓冲
        },
    )


# ---------- 接口 2：健康检查 ----------
@app.get("/health")
def health():
    """服务活着没、库里有多少块、有几份简历。
    /health 必须永远不崩：它崩了前端就判定整个后端挂了。"""
    chunks = collection.count()
    try:
        n_resumes = len(rag_core.list_resumes())
        orphans = rag_core.orphan_chunk_count()
    except Exception as e:
        return {"status": "degraded", "chunks": chunks, "resumes": 0,
                "error": f"{type(e).__name__}: {e}"}
    return {"status": "ok", "chunks": chunks, "resumes": n_resumes,
            "orphan_chunks": orphans}


# ---------- 接口 2b：清理无主块 ----------
@app.delete("/orphans")
def remove_orphans():
    """清掉 build_db.py 留下的、没有简历归属的老块。"""
    n = rag_core.clear_orphans()
    return {"message": "已清理", "deleted": n}


# ---------- 接口 3：上传简历 ----------
@app.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    resume_name: str = Form(...),
):
    """上传一份简历（纯文本 .txt），自动切分 + 向量化 + 入库，同时归档原文。

    - file：上传的文件
    - resume_name：简历显示名（如"张三"）；同名会被拒绝，避免误覆盖
    """
    # 1. 校验文件类型
    if not (file.filename or "").endswith(".txt"):
        raise HTTPException(status_code=400, detail="目前只支持 .txt 格式")

    # 2. 同名检查 —— 名字不能当唯一标识，同名一律拒绝，让用户先删再传
    existing = rag_core.find_by_name(resume_name)
    if existing:
        old_ids = "、".join(e["resume_id"] for e in existing)
        raise HTTPException(
            status_code=409,
            detail=f"已存在同名简历「{resume_name}」（{old_ids}）。为避免覆盖同名不同人，请先删除，或换个名字上传。",
        )

    # 3. 读文件内容
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件编码必须是 UTF-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件是空的")

    # 4. 生成唯一 resume_id（时间戳 + 文件名）
    #    去掉扩展名：文件名常带 .txt，再加一次就成了 x.txt.txt
    stem = os.path.splitext(file.filename or "resume")[0]
    resume_id = f"{int(time.time())}-{stem}"

    # 5. 建库：原文归档 + 切分向量化入 Chroma
    try:
        n_chunks = rag_core.build_resume(text, resume_id, resume_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"建库失败：{e}")

    return {
        "message": "上传成功",
        "resume_id": resume_id,
        "resume_name": resume_name,
        "chunks": n_chunks,
    }


# ---------- 接口 4：列出所有简历 ----------
@app.get("/resumes")
def get_resumes():
    """列出库里所有简历（含原文是否已归档）。"""
    try:
        return rag_core.list_resumes()
    except Exception as e:
        # 500 里带上真实原因，前端直接把这句话显示出来，不用再猜
        raise HTTPException(
            status_code=500,
            detail=f"读取简历列表失败：{type(e).__name__}: {e}",
        )


# ---------- 接口 5：下载原文 ----------
@app.get("/resumes/{resume_id}/download")
def download_resume(resume_id: str):
    """按 resume_id 取回简历原文。"""
    raw = rag_core.get_raw(resume_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="没找到这份简历的原文")
    return PlainTextResponse(
        content=raw,
        headers={"Content-Disposition": f'attachment; filename="{resume_id}.txt"'},
    )


# ---------- 接口 5b：简历正文（给侧边抽屉的"全文"视图用）----------
@app.get("/resumes/{resume_id}/text")
def resume_text(resume_id: str):
    """返回原文的 JSON（不带下载头），前端直接渲染。"""
    raw = rag_core.get_raw(resume_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="没找到这份简历的原文")
    return {"resume_id": resume_id, "text": raw}


# ---------- 接口 5c：简历摘要（给侧边抽屉的"概要"视图用）----------
@app.get("/resumes/{resume_id}/summary")
def resume_summary(resume_id: str, refresh: bool = False):
    """摘要（150-260 字）。

    第一次调用会现生成并缓存到 storage/summaries/，
    之后直接读缓存，不再调用大模型 —— 侧边抽屉秒开。
    refresh=true 时强制重新生成。
    """
    raw = rag_core.get_raw(resume_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="没找到这份简历的原文")

    cache = os.path.join(summary_core.SUMMARY_DIR, f"{resume_id}.txt")
    if not refresh and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return {"resume_id": resume_id, "summary": f.read(), "cached": True}

    try:
        s = analyzer.summarize(raw)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"摘要生成失败：{type(e).__name__}: {e}")

    with open(cache, "w", encoding="utf-8") as f:
        f.write(s)
    return {"resume_id": resume_id, "summary": s, "cached": False}


# ---------- 接口 5d：综合评价打分 ----------
class ScoreRequest(BaseModel):
    resume_id: str
    target_role: str = "大模型应用开发工程师"


@app.post("/score")
def score(req: ScoreRequest):
    """按四个维度给单份简历打分，返回分数 + 依据 + 优缺点 + 改进建议。"""
    raw = rag_core.get_raw(req.resume_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="没找到这份简历的原文")
    try:
        return analyzer.score_resume(raw, req.target_role)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"评分失败：{type(e).__name__}: {e}")


# ---------- 接口 5e：JD 匹配打分 ----------
class MatchRequest(BaseModel):
    resume_id: str
    jd: str


@app.post("/match")
def match(req: MatchRequest):
    """简历 vs 岗位 JD 的逐项对比 + 匹配度。"""
    if not req.jd.strip():
        raise HTTPException(status_code=400, detail="JD 不能为空")
    raw = rag_core.get_raw(req.resume_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="没找到这份简历的原文")
    try:
        return analyzer.match_resume(raw, req.jd)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"匹配分析失败：{type(e).__name__}: {e}")



# ---------- 接口 6：删除简历 ----------
@app.delete("/resumes/{resume_id}")
def remove_resume(resume_id: str):
    """删除一份简历：Chroma 里的块 + storage 里的原文。"""
    n = rag_core.delete_resume(resume_id)
    if n == 0:
        raise HTTPException(status_code=404, detail="没找到这份简历")
    return {"message": "已删除", "chunks_deleted": n}


# ---------- 接口 7：全局分析（跨简历比较）----------
@app.post("/compare")
def compare_resumes(req: ChatRequest):
    """全局问题：跨所有简历做比较/汇总。

    与 /chat 的区别：
    - /chat 走向量检索，只捞最相关的几块 → 适合"某个人的某个问题"
    - /compare 把所有简历摘要一起喂给大模型 → 适合"这批人的共同点/谁最匹配"
    """
    return summary_core.compare(req.question)


# ---------- 前端页面（必须放在所有路由之后）----------
# 顺序很重要：挂载 /static 在前，再给 / 一个返回 index.html 的路由。
# 如果先挂 "/"，后面所有路由都会被它吞掉（todo-api 踩过的坑）。
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
