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

def query_pod(pod_id):
    # Let's query as much info as we can
    query = f"""
    query {{
      pod(input: {{ podId: "{pod_id}" }}) {{
        id
        name
        imageName
        machineId
        machine {{
          podHostId
        }}
      }}
    }}
    """
    url = "https://api.runpod.io/graphql"
    r = requests.post(url, json={"query": query}, headers=HEADERS)
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    query_pod("hl5unqg9811jn9")
