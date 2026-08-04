import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# 1. 建立连接
conn = psycopg2.connect(host="127.0.0.1", port=5433, user="postgres", password="test123", dbname="postgres")
cur = conn.cursor()

# 2. 激活 vector 扩展
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# 3. 注册 vector 数据类型
register_vector(conn)

# 4. 删除旧表（避免之前的 384 维旧表结构干扰）并重新创建 512 维向量表
cur.execute("DROP TABLE IF EXISTS documents;")
cur.execute("CREATE TABLE documents (id SERIAL PRIMARY KEY, content TEXT, embedding vector(512))")
conn.commit()

# 5. 向量化 + 存入
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

# 6. 语义检索（余弦距离 <=> 越小越相似）
query = "什么是向量数据库？"
qe = model.encode(query).tolist()
cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents ORDER BY embedding <=> %s::vector LIMIT 3
""", (qe, qe))

print(f"🔍 '{query}'")
for i, (content, sim) in enumerate(cur.fetchall(), 1):
    print(f"  {i}. [{sim:.4f}] {content}")

# 7. 清理并关闭连接
cur.execute("DROP TABLE IF EXISTS documents")
conn.commit()
cur.close()
conn.close()