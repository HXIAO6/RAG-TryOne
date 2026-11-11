import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =========================
# 1. 模拟工业文档
# =========================

documents = [
    {
        "id": "doc1",
        "file": "industrial_standard.pdf",
        "page": 12,
        "text": "GB/T 12345-2024 第5.2条规定：设备最高工作压力不得超过10MPa。"
    },
    {
        "id": "doc2",
        "file": "industrial_standard.pdf",
        "page": 13,
        "text": "GB/T 12345-2024 第5.3条规定：设备测试压力应达到15MPa。"
    },
    {
        "id": "doc3",
        "file": "industrial_standard.pdf",
        "page": 18,
        "text": "GB/T 12345-2024 第6.1条规定：设备工作温度范围为-10℃至50℃。"
    },
    {
        "id": "doc4",
        "file": "safety_manual.pdf",
        "page": 8,
        "text": "设备运行过程中应做好安全防护，并定期检查压力表。"
    }
]


# =========================
# 2. 中文 + 标准号分词
# =========================

def tokenize(text: str) -> list[str]:
    """
    中文文本分词。
    尽量保留 GB/T、2024、5.2 这种标准号信息。
    """

    parts = re.findall(
        r"[A-Za-z]+(?:/[A-Za-z]+)?"
        r"|\d+(?:-\d+)?(?:\.\d+)?"
        r"|[\u4e00-\u9fff]+",
        text
    )

    tokens = []

    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9/.-]+", part):
            tokens.append(part)
        else:
            tokens.extend(
                word.strip()
                for word in jieba.lcut(part)
                if word.strip()
            )

    return tokens


# =========================
# 3. 建立 BM25
# =========================

tokenized_documents = [
    tokenize(doc["text"])
    for doc in documents
]

bm25 = BM25Okapi(tokenized_documents)


# =========================
# 4. 建立 Embedding
# =========================

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

document_texts = [
    doc["text"]
    for doc in documents
]

document_embeddings = model.encode(
    document_texts,
    normalize_embeddings=True
)


# =========================
# 5. BM25 检索
# =========================

def bm25_search(query: str, top_k: int = 3):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "score": float(scores[index]),
            "document": documents[index]
        })

    return results


# =========================
# 6. Embedding 检索
# =========================

def dense_search(query: str, top_k: int = 3):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # 因为向量已经归一化，
    # 点积就等价于余弦相似度
    scores = document_embeddings @ query_embedding

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "score": float(scores[index]),
            "document": documents[index]
        })

    return results


# =========================
# 7. 打印结果
# =========================

def print_results(title: str, results):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        doc = result["document"]

        print(f"\nTop {rank}")
        print(f"score: {result['score']:.4f}")
        print(f"文件: {doc['file']}")
        print(f"页码: {doc['page']}")
        print(f"内容: {doc['text']}")


# =========================
# 8. 两种查询
# =========================

queries = [
    {
        "name": "标准号查询",
        "query": "GB/T 12345-2024 5.2"
    },
    {
        "name": "语义改写查询",
        "query": "2024版标准第5.2条对设备最大允许工作压力有什么要求？"
    }
]


# =========================
# 9. 执行实验
# =========================

for item in queries:

    query_name = item["name"]
    query = item["query"]

    print()
    print("\n" + "#" * 60)
    print(query_name)
    print("查询：", query)
    print("分词：", tokenize(query))
    print("#" * 60)

    bm25_results = bm25_search(query, top_k=3)

    dense_results = dense_search(query, top_k=3)

    print_results(
        "BM25 检索结果",
        bm25_results
    )

    print_results(
        "Embedding 检索结果",
        dense_results
    )