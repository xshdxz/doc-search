"""
pgvector_demo.py — 用 pgvector 做向量存储和检索
Chroma vs pgvector：Chroma轻量/即装即用，pgvector支持ACID事务/适合生产环境
"""
from pgvector.psycopg2 import register_vector
import psycopg2
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="test123", dbname="postgres")
register_vector(conn)
cur = conn.cursor()

# 建向量表（384维 = BGE-small 的输出维度）
cur.execute("CREATE TABLE IF NOT EXISTS documents (id SERIAL PRIMARY KEY, content TEXT, embedding vector(384))")
conn.commit()

# 向量化 + 存入
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
docs = [
    "pgvector 是 PostgreSQL 的向量扩展，支持高效的向量相似度搜索",
    "Chroma 是一个轻量级的向量数据库，专为 LLM 应用设计",
    "余弦相似度是衡量两个向量方向相似程度的常用方法",
]
embs = model.encode(docs)
for doc, emb in zip(docs, embs):
    cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (doc, emb.tolist()))
conn.commit()

# 语义检索（余弦距离 <=> 越小越相似）
query = "什么是向量数据库？"
qe = model.encode(query).tolist()
cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents ORDER BY embedding <=> %s::vector LIMIT 3
""", (qe, qe))

print(f"🔍 '{query}'")
for i, (content, sim) in enumerate(cur.fetchall(), 1):
    print(f"  {i}. [{sim:.4f}] {content}")

cur.execute("DROP TABLE IF EXISTS documents")
conn.commit()
cur.close()
conn.close()