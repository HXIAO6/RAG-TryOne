import json

with open("output/result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data)
print(data["file"])
print(data["page"])
print(data["text"])