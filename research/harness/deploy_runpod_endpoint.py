import os, json, base64, requests
from dotenv import load_dotenv
load_dotenv("research/.env")
key = os.environ["RUNPOD_API_KEY"]
H = {"Authorization": f"Bearer {key}"}

handler_path = os.path.join(os.path.dirname(__file__), "runpod_handler.py")
with open(handler_path, "rb") as f:
    handler_bytes = f.read()
b64 = base64.b64encode(handler_bytes).decode()
cmd = f"pip install --no-cache-dir runpod && echo {b64} | base64 -d > /app/handler.py && python3 -u /app/handler.py"
docker_args = json.dumps({"cmd": ["bash", "-c", cmd]})

# Delete old endpoints
q = '''query { myself { endpoints { id name } } }'''
r = requests.post('https://api.runpod.io/graphql', json={'query': q}, headers=H).json()
for ep in r.get('data',{}).get('myself',{}).get('endpoints',[]):
    if ep['name'] not in ['mres-test-simple-v2-ep']:
        epid = ep['id']
        requests.post('https://api.runpod.io/graphql', json={'query': f'mutation {{ deleteEndpoint(id: "{epid}") }}'}, headers=H)
        print(f'Deleted {ep["name"]}')

tpl_name = "mres-reason-v2"
tpl = f'''
mutation {{
  saveTemplate(input: {{
    name: "{tpl_name}",
    imageName: "vllm/vllm-omni:cosmos3",
    dockerArgs: {json.dumps(docker_args)},
    containerDiskInGb: 30,
    env: [
      {{key: "RUNPOD_SKIP_AUTO_SYSTEM_CHECKS", value: "true"}},
      {{key: "HF_TOKEN", value: "{{{{ RUNPOD_SECRET_HF_TOKEN }}}}"}},
      {{key: "MODEL_ID", value: "nvidia/Cosmos-Reason1-7B"}},
      {{key: "MAX_MODEL_LEN", value: "32768"}}
    ],
    isServerless: true, startSsh: false, isPublic: false,
    readme: "", ports: "", volumeInGb: 0
  }}) {{ id }}
}}
'''
resp = requests.post("https://api.runpod.io/graphql", json={"query": tpl}, headers=H).json()
if "errors" in resp:
    print("Template error:", json.dumps(resp["errors"]))
    exit(1)
tpl_id = resp["data"]["saveTemplate"]["id"]
print(f"Template: {tpl_id}")

ep = f'''
mutation {{
  saveEndpoint(input: {{
    name: "{tpl_name}-ep",
    templateId: "{tpl_id}",
    gpuIds: "AMPERE_80,HOPPER_141",
    idleTimeout: 10,
    scalerType: "QUEUE_DELAY", scalerValue: 4,
    workersMin: 1, workersMax: 1, gpuCount: 1
  }}) {{ id }}
}}
'''
resp = requests.post("https://api.runpod.io/graphql", json={"query": ep}, headers=H).json()
if "errors" in resp:
    print("Endpoint error:", json.dumps(resp["errors"]))
    exit(1)
ep_id = resp["data"]["saveEndpoint"]["id"]
print(f"Endpoint: {ep_id}")

r = requests.post(f'https://api.runpod.ai/v2/{ep_id}/run',
    headers={'Authorization': f'Bearer {key}'},
    json={"input": {"endpoint": "models", "payload": {}}}, timeout=5)
print("Job:", r.json())
