from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


sentences = [
    "设备最大可以承受多大的压力？",
    "设备最高允许工作压力是多少？",
    "设备工作温度是多少？"
]


embeddings = model.encode(sentences)


for sentence, embedding in zip(sentences, embeddings):
    print("文本:", sentence)
    print("向量维度:", len(embedding))
    print("向量前5个值:", embedding[:5])
    print()