import os
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


def chunk_text(text, size=200):
    """切分：按段落切，段落太长就按字数硬切。纯代码逻辑，不需要模型。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= size:
            chunks.append(p)
        else:
            for i in range(0, len(p), size):
                chunks.append(p[i:i + size])
    return chunks


def embed(texts):
    """向量化：调硅基流动的 embedding 模型，把文字列表转成向量列表。"""
    resp = client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts,
    )
    return [item.embedding for item in resp.data]


# 读简历 → 切分 → 向量化 → 存库
with open("resume.txt", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)
print(f"切成了 {len(chunks)} 块：")
for i, c in enumerate(chunks):
    print(f"  [{i}] {c[:40]}...")

vectors = embed(chunks)
print(f"向量化完成，每块向量 {len(vectors[0])} 维")

collection.add(
    documents=chunks,
    embeddings=vectors,
    ids=[f"chunk-{i}" for i in range(len(chunks))],
)
print(f"建库完成，库中共 {collection.count()} 块")