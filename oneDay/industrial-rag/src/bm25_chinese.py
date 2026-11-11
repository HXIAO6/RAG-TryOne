import jieba
from rank_bm25 import BM25Okapi


documents = [
    "GB/T 12345-2024 第5.2条规定设备最高工作压力为10MPa。",
    "GB/T 12345-2024 第6.1条规定设备工作温度为50℃。",
    "GB/T 99999-2023 第3条规定设备安全要求。"
]


tokenized_documents = [
    jieba.lcut(document)
    for document in documents
]


bm25 = BM25Okapi(tokenized_documents)


query = "压力有什么规定？"

tokenized_query = jieba.lcut(query)

scores = bm25.get_scores(tokenized_query)


for document, score in zip(documents, scores):
    print(score, document)