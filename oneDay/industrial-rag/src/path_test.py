from pathlib import Path

data_dir = Path("data")

for file_path in data_dir.glob("*.pdf"):
    print(file_path)