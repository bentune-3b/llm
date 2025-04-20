import json

def extract_unique_sources_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_sources = set()

    for item in data:
        source = item.get("source")
        if source:
            unique_sources.add(source)

    return sorted(unique_sources)

if __name__ == "__main__":
    file_path = "model/data/raw/dev_data.json"
    sources = extract_unique_sources_from_file(file_path)
    print("Unique sources:")
    for src in sources:
        print(src)

