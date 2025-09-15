import json

def clean_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for entry in data:
        if "emotion" in entry and entry["emotion"] and isinstance(entry["emotion"], str):
            cleaned_data.append(entry)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned {file_path}. Valid entries: {len(cleaned_data)}")

# Run cleaning on all files
clean_json_file("dataset/train.json")
clean_json_file("dataset/validation.json")
clean_json_file("dataset/test.json")
