"""
main.py — 本地文档语义检索
运行方式：
  python main.py                → search 语义检索（默认）
  python main.py chunk-compare  → 分块策略对比实验
"""
import os
import sys
import chromadb
from chromadb.utils import embedding_functions


# ===== 初始化 =====
# SentenceTransformerEmbeddingFunction：用本地 BGE 模型做文本→向量转换
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"                    # 中文优化，仅133MB
)
client = chromadb.PersistentClient(path="./chroma_db")     # 数据持久化到本地


def get_documents() -> list[dict]:
    """从 docs/ 文件夹读取文档，没有则用内置示例"""
    docs = []
    if os.path.exists("docs"):
        for filename in os.listdir("docs"):
            if filename.endswith(".txt"):
                with open(f"docs/{filename}", "r", encoding="utf-8") as f:
                    docs.append({"content": f.read(), "source": filename})
    if not docs:
        docs = [
            {"content": "Python 是一种广泛使用的编程语言，以其简洁的语法和强大的库生态而闻名。", "source": "builtin"},
            {"content": "向量数据库专门用于存储和检索高维向量，常用于语义搜索和推荐系统。", "source": "builtin"},
            {"content": "DeepSeek 是深度求索公司开发的大语言模型，API 完全兼容 OpenAI SDK。", "source": "builtin"},
            {"content": "FastAPI 是一个现代 Python Web 框架，支持异步处理和自动生成 API 文档。", "source": "builtin"},
        ]
    return docs


# ================================================================
# mode=search: Chroma 语义检索
# ================================================================
def mode_search() -> None:
    """创建 Chroma Collection → 存入文档向量 → 语义查询"""
    collection = client.get_or_create_collection(
        name="my_docs",
        embedding_function=ef,
    )

    # 加载文档
    docs = get_documents()
    if collection.count() == 0:
        collection.add(
            documents=[d["content"] for d in docs],
            ids=[str(i) for i in range(len(docs))],
            metadatas=[{"source": d["source"]} for d in docs],
        )
        print(f"✅ 已存入 {len(docs)} 篇文档")

    # 交互式查询
    print(f"\n🔍 语义检索就绪（共 {collection.count()} 篇文档）")
    while True:
        query = input("\n搜索: ").strip()
        if query == "/exit":
            break
        results = collection.query(query_texts=[query], n_results=3)
        for i, (doc, dist, meta) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            print(f"  [{i+1}] 距离={dist:.4f} | {meta['source']}")
            print(f"      {doc[:100]}...")


# ================================================================
# mode=chunk-compare: 分块策略对比
# ================================================================
def naive_chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """滑动窗口分块——按固定字符数切分，overlap 让相邻 chunk 有重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap                       # overlap 让窗口有重叠
    return chunks


def simulate_retrieval(chunks: list[str], keywords: list[str]) -> int:
    """模拟检索命中率：包含任意关键词的 chunk 算命中"""
    return sum(1 for c in chunks if any(kw in c for kw in keywords))


def mode_chunk_compare() -> None:
    """对比 4 种分块配置的检索命中率"""
    sample = """人工智能的发展经历了多个阶段。从早期的符号主义到现代的深度学习，
    技术路线不断演进。大语言模型的出现标志着AI进入了一个新时代。向量数据库作为
    AI基础设施的重要组成部分，为检索增强生成提供了关键支持。RAG（检索增强生成）
    技术将信息检索与文本生成相结合，有效解决了大模型的幻觉问题。"""
    sample = sample * 5                                     # 放大文本量以便对比

    keywords = ["向量数据库", "RAG", "检索", "大语言模型"]
    configs = [
        (300, 0, "小chunk/无重叠"),
        (1000, 0, "大chunk/无重叠"),
        (300, 50, "小chunk/有重叠"),
        (1000, 200, "大chunk/有重叠 ✅推荐"),
    ]

    print(f"\n📊 分块策略对比（原文{len(sample)}字，关键词: {keywords}）")
    print(f"{'策略':<25} {'chunk数':<10} {'命中chunk':<10} {'覆盖率'}")
    print("-" * 60)
    for size, overlap, label in configs:
        chunks = naive_chunk(sample, size, overlap)
        hits = simulate_retrieval(chunks, keywords)
        coverage = hits / len(chunks) if chunks else 0
        print(f"{label:<25} {len(chunks):<10} {hits:<10} {coverage:.1%}")


# ===== 主入口 =====
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chunk-compare":
        mode_chunk_compare()
    else:
        mode_search()