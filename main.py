"""
main.py — 本地文档语义检索
==========================
两种模式：
  search        语义检索（默认）：把文档向量化 → 存 Chroma → 自然语言查询
  chunk-compare 分块策略对比：同一篇文档，不同分块参数，看命中率差异

核心概念：
  Embedding：把文本变成高维向量，语义相近的文本向量距离近
  Chroma：轻量级向量数据库，存向量 + 元数据，支持相似度检索
  Chunking：长文档必须切块，每块是检索的最小单位
"""

import sys
import os
import chromadb                                          # 轻量级向量数据库
from chromadb.utils import embedding_functions           # 内置 Embedding 模型封装

# ================================================================
# 共享组件：Embedding 模型 + 文档加载
# ================================================================

# Embedding 函数：负责把"文本"转成"向量"
# all-MiniLM-L6-v2 是一个轻量级英文模型（约 80MB），首次运行自动下载
# 将文本映射到 384 维向量空间，语义相近的文本向量距离也近
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"                                   # 384 维向量，速度快
)

def load_documents_from_folder(folder_path:str) -> list[dict]:
    """
    从文件夹读取所有 .txt 文件

    参数:
        folder_path: 文档文件夹路径
    返回:
        [{"id": "文件名(不含扩展名)", "text": "文件内容"}, ...]
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):                                           # 只处理文本文件
            with open(os.path.join(folder_path,filename),
                      "r",encoding="UTF-8") as f:
                docs.append({
                    "id" : filename.replace(".txt",""),             # 用文件名作为文档 ID
                    "text" : f.read()
                })
    return docs

def get_documents() -> list[dict]:
    """
    获取文档列表：优先从 docs/ 文件夹读，没有则用内置示例

    返回:
        [{"id": str, "text": str}, ...]
    """
    if os.path.exists("docs") and os.listdir("docs"):
        return load_documents_from_folder("docs")
    else:
        # 内置示例文档（6 条，覆盖不同技术主题）
        contents = [
            "Python 是一种解释型、面向对象的高级编程语言。它的设计哲学强调代码的可读性。",
            "FastAPI 是一个现代的、高性能的 Python Web 框架，基于标准 Python 类型提示构建。",
            "LangChain 是一个用于构建 LLM 应用的框架，提供了 Chain、Agent、Memory 等核心组件。",
            "向量数据库用于存储和检索高维向量，是 RAG 系统的核心组件。Chroma 是一个轻量级的向量数据库。",
            "Docker 是一个开源的容器化平台，用于将应用程序及其依赖打包到轻量级、可移植的容器中。",
            "PostgreSQL 是一个功能强大的开源关系型数据库，支持扩展如 pgvector 用于向量检索。",
        ]
        return [{"id":f"doc_{i}","text":t} for i,t in enumerate(contents)]

# ================================================================
# 模式 1：语义检索（默认）
# ================================================================
def mode_search() -> None:
    """
    语义检索模式

    流程：
    1. 创建 Chroma 客户端 + Collection（类似"建表"）
    2. 把文档向量化后存入 Collection
    3. 用自然语言查询，返回最相关的文档

    关键概念：
    - Collection：向量数据库中的"表"，一个项目通常一个 Collection
    - query 返回的 distance：向量之间的"距离"，越小越相似
    """
    print("🔍 语义检索模式\n")

    # Step 1: 创建 Chroma 客户端（内存模式，数据不持久化）
    client = chromadb.Client()

    # Step 2: 创建 Collection
    # name: Collection 的唯一标识
    # embedding_function: 存入/查询时自动把文本转成向量
    collection = client.create_collection(
        name = "my_docs",
        embedding_function=ef,                                  # Chroma 会自动调用它来向量化文本
    )

    # Step 3: 加载文档并存入
    docs = get_documents()
    collection.add(
        ids=[d["id"] for d in docs],                                # 每条文档的唯一 ID（用于更新/删除）
        documents=[d["text"] for d in docs],                        # 文档正文（会被自动 Embedding）
        metadatas=[{"source" : d["id"]} for d in docs],             # 附加信息（来源、作者等，可任意扩展）
    )
    print(f"✅ 已存入{collection.count()}条文档\n")

    # Step 4: 语义检索
    # 查询时 Chroma 自动把查询文本向量化，然后找最相似的文档
    queries = [
        "什么是容器化技术？",                                        # 应该命中 Docker 那条
        "怎么构建 AI 应用？",                                        # 应该命中 LangChain 那条
        "Python Web 框架有哪些？",                                   # 应该命中 FastAPI 那条
    ]
    for query in queries:
        results = collection.query(
            query_texts=[query],                                    # 查询文本（可以是多个）
            n_results = 2                                           # 返回 Top-2 最相关的
        )
        print(f"🔍 查询: {query}")
        for i,doc in enumerate(results["documents"][0]):                    # documents[0] = 第一个查询的结果
            d = results["distances"][0][i]                                  # 距离越小 = 越相似
            print(f" Top-{i+1} (距离: {d:.4f}): {doc[:80]}...")
        print()


# ================================================================
# 模式 2：分块策略对比实验（对应学习路线 7/26）
# ================================================================
def mode_chunk_compare() -> None:
    """
    分块策略对比实验

    为什么需要分块？
    - LLM 的 context window 有限（几万 token），不能一次塞整本书
    - 分块后，只检索最相关的几个块给 LLM
    - 分块的好坏直接影响 RAG 的检索质量

    实验设计：
    对比 4 种配置（chunk_size × overlap）的检索命中率
    - chunk_size: 每块多大（太小信息不完整，太大检索不精准）
    - overlap: 相邻块的重叠量（防止关键信息恰好落在分块边界上）
    """
    print("🔬 分块策略对比实验\n")

    # 把所有文档拼成一篇长文本（模拟真实的长文档）
    docs = get_documents()
    long_text = "\n\n".join(d["text"] for d in docs)
    print(f"📄 原文长度: {len(long_text)} 字符\n")

    def naive_chunk(text:str,chunk_size:int,overlap:int) -> list[dict]:
        """
        滑动窗口分块算法

        参数:
            text: 原始文本
            chunk_size: 每块最大字符数
            overlap: 相邻块重叠的字符数
        返回:
            文本块列表

        示例: text="ABCDEFGH", chunk_size=4, overlap=2
              → ["ABCD", "CDEF", "EFGH"]
        """
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += (chunk_size - overlap)                                     # 步长 = chunk_size - overlap
        return chunks

    def simulate_retrieval(chunks:str,keywords:list[str]) -> dict:
        """
        模拟检索：统计包含任意关键词的块数

        这不是真正的向量检索，只是用来对比不同分块策略下
        关键词落在多少块里——命中率越高，越可能被检索到
        """
        hits = sum(1 for c in chunks if any(kw in c for kw in keywords))
        return {
            "total":len(chunks),
            "hits":hits,
            "rate":f"{hits / len(chunks) * 100:.1f}%" if chunks else "N/A"
        }

    # 4 种分块策略
    configs = [
        (300, 0, "小块无重叠"),                                  # 太碎，信息不完整
        (300, 50, "小块有重叠"),                                 # 碎但有冗余
        (1000, 0, "大块无重叠"),                                 # 边界信息可能丢失
        (1000, 200, "大块有重叠（推荐）✅"),                                  # RAG 最佳实践
    ]
    keywords = ["向量", "Embedding", "检索", "Python", "数据库"]

    print(f"🔑 查询关键词: {keywords}\n")
    for size,overlap,label in configs:
        chunks = naive_chunk(long_text,size,overlap)
        r = simulate_retrieval(chunks,keywords)
        print(f"  {label}: chunk_size={size}, overlap={overlap}")
        print(f"    → {r['total']} 块, 命中 {r['hits']} 块, 命中率 {r['rate']}")


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    # 从命令行参数获取模式，默认 "search"
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    if mode == "chunk-compare":
        mode_chunk_compare()
    else:
        mode_search()