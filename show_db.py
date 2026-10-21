"""查看向量库里存了什么——原文、编号，以及（可选的）向量预览。
用法：python show_db.py
"""
import chromadb

# 连上库
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("resume")

data = collection.get(include=["documents", "embeddings"])

docs = data["documents"]
ids = data["ids"]
embeds = data.get("embeddings")

print(f"库中共 {len(ids)} 块\n")
print("=" * 60)

for i, (_id, doc) in enumerate(zip(ids, docs)):
    print(f"\n【{i}】 id={_id}  ({len(doc)}字)")
    print("-" * 60)
    print(doc)
    if embeds is not None:
        # 向量太长，只显示前 5 个数感受一下
        vec = embeds[i]
        print(f"\n  → 向量维度 {len(vec)}，前5个值：{[round(x, 3) for x in vec[:5]]}")

print("\n" + "=" * 60)
print("提示：documents 是原文（人看），embeddings 是向量（机器算相似度）——两者一一对应。")
