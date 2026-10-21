import os

def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

def main():
    cache_path = "/runpod-volume/huggingface-cache"
    print("Checking HuggingFace cache directory on RunPod:")
    if not os.path.exists(cache_path):
        print(f"Path '{cache_path}' does not exist.")
        return
        
    print(f"Path '{cache_path}' exists.")
    # List files/folders
    print("Directory contents:")
    for item in os.listdir(cache_path):
        item_path = os.path.join(cache_path, item)
        size_bytes = get_dir_size(item_path)
        size_gb = size_bytes / (1024 ** 3)
        print(f"- {item}/: {size_gb:.2f} GB")

if __name__ == "__main__":
    main()
