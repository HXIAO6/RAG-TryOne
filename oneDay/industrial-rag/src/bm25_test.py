from rank_bm25 import BM25Okapi


documents = [
    "GB/T 12345-2024 pressure requirement",
    "GB/T 12345-2024 temperature requirement",
    "GB/T 99999-2023 safety requirement",
]


tokenized_documents = [
    document.split()
    for document in documents
]


bm25 = BM25Okapi(tokenized_documents)


query = "GB/T 12345-2024 pressure"

tokenized_query = query.split()


scores = bm25.get_scores(tokenized_query)


for document, score in zip(documents, scores):
    print(score, document)