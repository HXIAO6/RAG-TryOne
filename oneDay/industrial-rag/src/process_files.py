from pathlib import Path
import json


def read_text(file_path: Path) -> str:
    if file_path.name == "b.txt":
        raise ValueError("模拟文件解析失败")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_result(file_path: Path, text: str, output_dir: Path) -> None:
    """把处理结果保存成 JSON。"""
    result = {
        "file": file_path.name,
        "text": text,
        "status": "success"
    }

    output_path = output_dir / f"{file_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )


def main() -> None:
    """批量处理 data 目录中的 TXT 文件。"""
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "output"

    output_dir.mkdir(exist_ok=True)

    failed_files: list[dict] = []

    for file_path in data_dir.glob("*.txt"):
        try:
            text = read_text(file_path)
            save_result(file_path, text, output_dir)

            print(f"成功: {file_path.name}")

        except Exception as e:
            print(f"失败: {file_path.name}")

            failed_files.append({
                "file": file_path.name,
                "error": str(e)
            })

    error_path = output_dir / "errors.json"

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(
            failed_files,
            f,
            ensure_ascii=False,
            indent=4
        )


if __name__ == "__main__":
    main()