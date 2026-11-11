import re

import jieba
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


app = FastAPI()


class SearchRequest(BaseModel):
    query: str


documents = [
    {
        "file": "industrial_standard.pdf",
        "page": 12,
        "text": "GB/T 12345-2024 第5.2条规定：设备最高工作压力不得超过10MPa。"
    },
    {
        "file": "industrial_standard.pdf",
        "page": 13,
        "text": "GB/T 12345-2024 第5.3条规定：设备测试压力应达到15MPa。"
    },
    {
        "file": "industrial_standard.pdf",
        "page": 18,
        "text": "GB/T 12345-2024 第6.1条规定：设备工作温度范围为-10℃至50℃。"
    },
    {
        "file": "safety_manual.pdf",
        "page": 8,
        "text": "设备运行过程中应做好安全防护，并定期检查压力表。"
    }
]


def tokenize(text: str) -> list[str]:
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


tokenized_documents = [
    tokenize(doc["text"])
    for doc in documents
]

bm25 = BM25Okapi(tokenized_documents)


@app.get("/")
def home():
    return {
        "message": "Industrial RAG Search API"
    }


@app.post("/search")
def search(request: SearchRequest):
    query_tokens = tokenize(request.query)

    scores = bm25.get_scores(query_tokens)

    indices = np.argsort(scores)[::-1][:5]

    results = []

    for index in indices:
        doc = documents[index]

        results.append({
            "score": float(scores[index]),
            "file": doc["file"],
            "page": doc["page"],
            "text": doc["text"]
        })

    return {
        "query": request.query,
        "results": results
    }