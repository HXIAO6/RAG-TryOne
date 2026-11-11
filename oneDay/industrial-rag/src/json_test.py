import json

documents = [
    {
        "file": "manual.pdf",
        "page": 1,
        "text": "设备名称：工业压缩机"
    },
    {
        "file": "manual.pdf",
        "page": 2,
        "text": "最高工作压力：10MPa"
    },
    {
        "file": "manual.pdf",
        "page": 3,
        "text": "工作温度：-10℃到50℃"
    }
]

with open("output/documents.json", "w", encoding="utf-8") as f:
    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=4
    )

print("保存成功")