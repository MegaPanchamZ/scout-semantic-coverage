import os
import requests
import sys
import json
import time

API_KEY = os.environ.get("RUNPOD_API_KEY")
if not API_KEY:
    print("Error: RUNPOD_API_KEY not set in environment.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def run_graphql(query, variables=None):
    url = "https://api.runpod.io/graphql"
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"GraphQL HTTP Error {response.status_code}: {response.text}")
    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL response errors: {data['errors']}")
    return data["data"]

def create_network_volume(name, size_gb, data_center_id):
    url = "https://rest.runpod.io/v1/networkvolumes"
    # Check if it already exists first
    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            vols_data = res.json()
            volumes = vols_data if isinstance(vols_data, list) else vols_data.get("networkVolumes", [])
            for vol in volumes:
                if vol.get("name") == name and vol.get("dataCenterId") == data_center_id:
                    print(f"Volume '{name}' already exists in {data_center_id} (ID: {vol['id']}). Reusing it.")
                    return vol
    except Exception as e:
        print(f"Note: failed to query existing volumes: {e}")

    payload = {
        "name": name,
        "size": size_gb,
        "dataCenterId": data_center_id
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code in (200, 201, 202):
        return response.json()
    elif response.status_code == 409:
        print(f"Volume '{name}' might already exist (409 conflict).")
        try:
            res = requests.get(url, headers=HEADERS)
            if res.status_code == 200:
                vols_data = res.json()
                volumes = vols_data if isinstance(vols_data, list) else vols_data.get("networkVolumes", [])
                for vol in volumes:
                    if vol.get("name") == name and vol.get("dataCenterId") == data_center_id:
                        return vol
        except Exception:
            pass
        return None
    else:
        print(f"Failed to create network volume (status {response.status_code}): {response.text}")
        sys.exit(1)

def terminate_pod(pod_id):
    mutation = """
    mutation TerminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    print(f"Terminating pod '{pod_id}'...")
    try:
        run_graphql(mutation, variables={"input": {"podId": pod_id}})
        print("Termination mutation sent successfully.")
    except Exception as e:
        print(f"Note: failed to terminate pod {pod_id}: {e}")

# We will implement a simplified script that primarily generates the correct payload and logs it.
def main():
    # Terminate the running caching pod
    terminate_pod("zdxerzdojtejni")
    
    print("Fetching lowest priced GPU types suitable for 32GB VRAM (Cosmos 3 Nano)...")
    query = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        communityPrice
      }
    }
    """
    data = run_graphql(query)
    
    suitable_gpus = []
    print("\nAll GPUs >= 32GB memory:")
    for gpu in data["gpuTypes"]:
        if gpu["memoryInGb"] >= 32:
            price = gpu.get("communityPrice")
            print(f"- {gpu['displayName']} ({gpu['memoryInGb']}GB): Price: {price}, ID: {gpu['id']}")
            if price is not None:
                suitable_gpus.append({
                    "id": gpu["id"],
                    "displayName": gpu["displayName"],
                    "price": price,
                    "vram": gpu["memoryInGb"]
                })
            
    # Sort by price
    suitable_gpus.sort(key=lambda x: x["price"])
    if not suitable_gpus:
        print("No suitable GPUs found.")
        sys.exit(1)
        
    cheapest_gpu = suitable_gpus[0]
    print(f"\nCheapest suitable GPU: {cheapest_gpu['displayName']} ({cheapest_gpu['vram']}GB VRAM) @ ${cheapest_gpu['price']}/hr")
    
    # Use EU-RO-1 which has cheap RTX PRO 4500 (32GB) in High stock, and cheap A100/RTX5090/PRO6000 fallbacks.
    data_center_id = "EU-RO-1"
    
    vol_name = "cosmos3-weights-cache"
    print(f"Creating 200GB Network Volume '{vol_name}' in {data_center_id}...")
    volume = create_network_volume(vol_name, 200, data_center_id)
    if not volume:
        # Fallback if endpoint is different
        print("Network volume API failed, assuming local temporary storage for now.")
        vol_id = "mock-vol-123"
    else:
        vol_id = volume.get("id", "unknown-id")
        print(f"Network Volume ID: {vol_id}")
    
    print("\n--- Deploying Pod ---")
    pod_payload = {
        "cloudType": "COMMUNITY",
        "gpuCount": 1,
        "volumeInGb": 50,
        "containerDiskInGb": 50,
        "gpuTypeId": cheapest_gpu["id"],
        "name": "Cosmos3-Cache-Builder",
        "imageName": "vllm/vllm-omni:cosmos3",
        "dockerArgs": "bash -c 'apt-get update && apt-get install -y openssh-server && mkdir -p /var/run/sshd && mkdir -p ~/.ssh && echo \"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKn6UoktAUiZMT2+W8YWJ0swsA7kGjFkmUX6ntL4jV+e\" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && echo \"PermitRootLogin yes\" >> /etc/ssh/sshd_config && echo \"PubkeyAuthentication yes\" >> /etc/ssh/sshd_config && ssh-keygen -A && /usr/sbin/sshd -D'",
        "ports": "8000/http,22/tcp",
        "volumeMountPath": "/runpod-volume",
        "env": [
            {"key": "HF_TOKEN", "value": "{{ RUNPOD_SECRET_HF_TOKEN }}"},
            {"key": "HF_HUB_CACHE", "value": "/runpod-volume/huggingface-cache"},
            {"key": "SSH_PUBLIC_KEY", "value": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKn6UoktAUiZMT2+W8YWJ0swsA7kGjFkmUX6ntL4jV+e deb@Debbabashish"}
        ],
        "networkVolumeId": vol_id,
        "dataCenterId": data_center_id
    }
    
    mutation = """
    mutation PodCreate($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
      }
    }
    """
    
    deployed = False
    for cloud_type in ["COMMUNITY", "SECURE"]:
        print(f"\n====== Trying Cloud Type: {cloud_type} ======")
        pod_payload["cloudType"] = cloud_type
        
        for gpu in suitable_gpus:
            print(f"Attempting to deploy on GPU: {gpu['displayName']} ({gpu['vram']}GB VRAM) on {cloud_type}...")
            pod_payload["gpuTypeId"] = gpu["id"]
            
            try:
                result = run_graphql(mutation, variables={"input": pod_payload})
                print(f"Successfully deployed pod on {gpu['displayName']} ({cloud_type})!")
                print(json.dumps(result, indent=2))
                deployed = True
                break
            except Exception as e:
                err_msg = str(e)
                if "SUPPLY_CONSTRAINT" in err_msg or "no longer any instances available" in err_msg:
                    print(f"GPU {gpu['displayName']} is out of stock in {data_center_id} for {cloud_type}. Trying next...")
                else:
                    print(f"Pod deployment failed for {gpu['displayName']} ({cloud_type}): {e}")
                    print("Trying next...")
        if deployed:
            break

    if not deployed:
        print(f"\nFailed to deploy pod on any suitable GPU in stock in {data_center_id}.")
        sys.exit(1)
    
    print("\nScript setup and deployment phase complete.")

if __name__ == "__main__":
    main()
