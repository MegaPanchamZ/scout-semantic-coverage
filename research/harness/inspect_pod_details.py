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

def check_pod_details(pod_id):
    query = """
    query {
      myself {
        pods {
          id
          name
          imageName
          desiredStatus
          memoryInGb
          runtime {
            uptimeInSeconds
            ports {
              ip
              privatePort
              publicPort
            }
          }
        }
      }
    }
    """
    url = "https://api.runpod.io/graphql"
    r = requests.post(url, json={"query": query}, headers=HEADERS)
    res = r.json()
    if "errors" in res:
        print("Errors:", res["errors"])
    
    pods = res.get("data", {}).get("myself", {}).get("pods", [])
    for p in pods:
        if p["id"] == pod_id:
            print(json.dumps(p, indent=2))
            return
    print(f"Pod {pod_id} not found in myself.pods list.")

if __name__ == "__main__":
    check_pod_details("s7atpko0p8w2ga")
