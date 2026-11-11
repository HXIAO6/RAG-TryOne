from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


sentences = [
    "设备最大可以承受多大的压力？",
    "设备最高允许工作压力是多少？",
    "设备工作温度是多少？"
]


embeddings = model.encode(sentences)


similarity = cosine_similarity(embeddings)


print(similarity)