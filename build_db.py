import os
import re
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 客户端：连硅基流动，用来把文字转成向量
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# 向量库：本地存一个文件夹，不需要联网、不需要服务器
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("resume")


def chunk_text(text, size=500, overlap=80):
    """
    切分策略（相比上一版的三大改进）：
    1. 按标题聚合 —— 把 [xxx] 当章节起点，标题和它下面的内容合成一块，标题不再孤立
    2. 上下文增强 —— 每块都带上所属章节的标题前缀，单独看也知道它属于哪部分
    3. 重叠切分   —— 超长块按 size 硬切时，相邻块重叠 overlap 字，防止句子被拦腰切断
    """
    blocks = []
    current_title = ""
    current_lines = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 识别标题行，如 [个人优势]、[技能清单]、[项目经历]
        if re.match(r"^\[.+\]$", line):
            if current_lines:
                blocks.append((current_title, "\n".join(current_lines)))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_title, "\n".join(current_lines)))

    # 把每一节组装成块：加标题前缀；太长则重叠切分
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
                # 除了第一片，其余每片都补上标题，保证上下文完整
                chunks.append(piece if start == 0 else f"{title}\n{piece}")
                if start + size >= len(full):
                    break
                start += size - overlap
    return chunks


def embed(texts):
    """向量化：调硅基流动的 embedding 模型，把文字列表转成向量列表。"""
    resp = client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts,
    )
    return [item.embedding for item in resp.data]


if __name__ == "__main__":
    # 注意：这个脚本是"从零建库"的早期版本，不写 metadata、不存原文，
    # 建出来的块没有 resume_id，前端列表里看不到。
    # 现在建库请走页面上传（POST /upload），那个才是正路。
    import sys
    if len(sys.argv) < 2:
        sys.exit("用法：python build_db.py 简历文件.txt\n"
                 "（日常建库请在页面上传，别用这个脚本）")
    path = sys.argv[1]

    # 重建前清空旧集合，避免旧的碎片数据混在里面
    try:
        chroma.delete_collection("resume")
        print("已清空旧集合")
    except Exception:
        pass
    collection = chroma.get_or_create_collection("resume")

    # 读简历 → 切分 → 向量化 → 存库
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"\n切成了 {len(chunks)} 块：")
    for i, c in enumerate(chunks):
        preview = c[:45].replace("\n", " / ")
        print(f"  [{i:2d}] ({len(c):3d}字) {preview}")

    vectors = embed(chunks)
    print(f"\n向量化完成，每块向量 {len(vectors[0])} 维")

    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
    )
    print(f"✅ 建库完成，库中共 {collection.count()} 块")
