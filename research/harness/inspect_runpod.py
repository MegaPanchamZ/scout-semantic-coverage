import os
import requests
import json

API_KEY = os.environ.get("RUNPOD_API_KEY")
if not API_KEY:
    print("Error: RUNPOD_API_KEY not set in environment.")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def check_gpus():
    query = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        communityPrice
        securePrice
      }
    }
    """
    url = "https://api.runpod.io/graphql"
    r = requests.post(url, json={"query": query}, headers=HEADERS)
    res = r.json()
    if "errors" in res:
        print("Errors:", res["errors"])
    
    gpus = res.get("data", {}).get("gpuTypes", [])
    suitable = []
    for g in gpus:
        if g["memoryInGb"] >= 32:
            suitable.append(g)
            
    # sort by communityPrice
    suitable.sort(key=lambda x: x.get("communityPrice") or 999)
    print("Suitable GPUs (VRAM >= 32GB):")
    for g in suitable:
        print(f"- ID: {g['id']} | Name: {g['displayName']} | VRAM: {g['memoryInGb']}GB | Community Price: ${g['communityPrice']}/hr | Secure Price: ${g['securePrice']}/hr")

if __name__ == "__main__":
    check_gpus()
