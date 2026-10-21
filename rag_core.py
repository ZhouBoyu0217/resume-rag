"""
RAG 核心能力：切分、向量化、建库、原文存储。
被 build_db.py（命令行）和 main.py（接口）共同调用——抽出来的目的就是复用。

设计要点：
- 索引与原文分离：Chroma 存"切分后的块 + 向量"，storage/ 存"原文原件"
- resume_id 是唯一身份，resume_name 只做展示（允许重名，但上传时会提示冲突）
"""
import os
import re
import shutil
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("resume")

# 原文存放目录（档案柜）
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


# ---------------- 切分 ----------------
def chunk_text(text, size=500, overlap=80):
    """
    切分策略：
    1. 按标题聚合 —— 把 [xxx] 当章节起点，标题和它下面的内容合成一块
    2. 上下文增强 —— 每块带上所属章节标题前缀
    3. 重叠切分   —— 超长块硬切时相邻重叠，防止句子被拦腰切断
    """
    blocks = []
    current_title = ""
    current_lines = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\[.+\]$", line):
            if current_lines:
                blocks.append((current_title, "\n".join(current_lines)))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_title, "\n".join(current_lines)))

    chunks = []
    for title, body in blocks:
        prefix = f"{title}\n" if title else ""
        full = prefix + body
        if len(full) <= size:
            chunks.append(full)
        else:
            start = 0
            while start < len(full):
                piece = full[start:start + size]
                chunks.append(piece if start == 0 else f"{title}\n{piece}")
                if start + size >= len(full):
                    break
                start += size - overlap
    return chunks


# ---------------- 向量化 ----------------
def embed(texts):
    """把一组文字转成向量（建库和查询共用同一模型，保证同一坐标系）。"""
    resp = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts,
    )
    return [item.embedding for item in resp.data]


# ---------------- 同名检查 ----------------
def _all_metadatas():
    """取出库里所有块的元数据，过滤掉 None。

    build_db.py 早期建库时没写 metadata，老块写进去就是 None。
    这里统一过滤掉，否则一调 .get() 就 NoneType 崩，整个 /resumes 报 500。
    """
    raw = collection.get(include=["metadatas"]).get("metadatas") or []
    return [m for m in raw if isinstance(m, dict)]


def find_by_name(resume_name):
    """按名字查已有的简历（名字可能重复，所以返回列表）。"""
    all_meta = _all_metadatas()
    found = {}
    for m in all_meta:
        if m.get("resume_name") == resume_name:
            rid = m.get("resume_id")
            found[rid] = {"resume_id": rid, "resume_name": m.get("resume_name")}
    return list(found.values())


# ---------------- 建库（索引 + 原文双写）----------------
def save_raw(resume_id, text):
    """把原文存进 storage/，文件名就是 resume_id。"""
    path = os.path.join(STORAGE_DIR, f"{resume_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def build_resume(text, resume_id, resume_name):
    """把一份简历建成库：原文归档 + 切分向量化入 Chroma。返回块数。"""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    # ① 原文归档（档案柜）
    save_raw(resume_id, text)

    # ② 切分后的块入索引（索引库）
    vectors = embed(chunks)
    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=[f"{resume_id}-chunk-{i}" for i in range(len(chunks))],
        metadatas=[{"resume_id": resume_id, "resume_name": resume_name}] * len(chunks),
    )
    return len(chunks)


# ---------------- 读取原文 ----------------
def get_raw(resume_id):
    """按 resume_id 取回原文；不存在返回 None。"""
    path = os.path.join(STORAGE_DIR, f"{resume_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------- 删除（索引 + 原文同步删）----------------
def delete_resume(resume_id):
    """删除一份简历：Chroma 里的块 + storage 里的原文。返回删除的块数。"""
    existing = collection.get(where={"resume_id": resume_id})
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)

    raw_path = os.path.join(STORAGE_DIR, f"{resume_id}.txt")
    if os.path.exists(raw_path):
        os.remove(raw_path)
    return len(ids)


# ---------------- 列表 ----------------
def orphan_chunk_count():
    """统计"没有简历身份"的老块数量。

    早年直接跑 build_db.py 建出来的块 metadata 是空的、id 还是 chunk-0 这种，
    它们能参与检索，但没有任何简历归属 —— 前端列表里看不到它们。
    上传新简历前提示用户清掉，否则问答会混进这些"无主"内容。
    """
    raw = collection.get(include=["metadatas"]).get("metadatas") or []
    return sum(1 for m in raw if not (isinstance(m, dict) and m.get("resume_id")))


def clear_orphans():
    """删除所有没有 resume_id 的老块，返回删除数量。"""
    got = collection.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    dead = [i for i, m in zip(ids, metas)
            if not (isinstance(m, dict) and m.get("resume_id"))]
    if dead:
        collection.delete(ids=dead)
    return len(dead)


def list_resumes():
    """列出库里所有简历（按 resume_id 去重），并标记原文是否存在。"""
    all_meta = _all_metadatas()
    seen = {}
    for m in all_meta:
        rid = m.get("resume_id")
        if rid and rid not in seen:
            seen[rid] = {
                "resume_id": rid,
                "resume_name": m.get("resume_name", rid),
                "has_raw": os.path.exists(os.path.join(STORAGE_DIR, f"{rid}.txt")),
            }
    return list(seen.values())
