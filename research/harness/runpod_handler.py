import os
import subprocess
import time
import requests
import runpod

VOLUME_PATH = "/runpod-volume"
if os.path.isdir("/workspace/huggingface-cache"):
    VOLUME_PATH = "/workspace"
elif os.path.isdir("/runpod-volume/huggingface-cache"):
    VOLUME_PATH = "/runpod-volume"

LOG_FILE = os.path.join(VOLUME_PATH, "serverless_vllm_startup.log")
VLLM_LOG_FILE = os.path.join(VOLUME_PATH, "serverless_vllm_vllm.log")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write to log file {LOG_FILE}: {e}")


MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Cosmos-Reason1-7B")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")  # ensure it's set
PORT = os.environ.get("PORT", "8000")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "32768")

os.environ["HF_HUB_CACHE"] = os.path.join(VOLUME_PATH, "huggingface-cache")
log(f"Detected network volume path: {VOLUME_PATH}")
log(f"Set HF_HUB_CACHE to: {os.environ['HF_HUB_CACHE']}")
log(f"Model: {MODEL_ID}")


def start_vllm_server():
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--port", PORT,
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
    ]
    log(f"Executing command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        vllm_log = open(VLLM_LOG_FILE, "a")
        process = subprocess.Popen(cmd, env=env, stdout=vllm_log, stderr=vllm_log)
        log(f"vLLM process spawned with PID {process.pid}")
        return process
    except Exception as e:
        log(f"Error spawning vLLM process: {e}")
        raise


def wait_for_server(port):
    url = f"http://localhost:{port}/v1/models"
    log(f"Polling models endpoint: {url}")
    start_time = time.time()
    while True:
        if vllm_process.poll() is not None:
            log("vLLM process exited unexpectedly!")
            try:
                if os.path.exists(VLLM_LOG_FILE):
                    with open(VLLM_LOG_FILE, "r") as lf:
                        lines = lf.readlines()[-20:]
                    log("Last 20 lines of vLLM log:")
                    for line in lines:
                        log(f"  vLLM: {line.strip()}")
            except Exception as le:
                log(f"Failed to read vLLM logs: {le}")
            raise RuntimeError("vLLM serving process died during startup.")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                log("vLLM server is fully initialized and online!")
                return
        except requests.exceptions.RequestException:
            pass
        if time.time() - start_time > 1800:
            log("vLLM startup timed out (1800s exceeded).")
            raise RuntimeError("vLLM startup timed out.")
        time.sleep(5)


try:
    log("Initializing server...")
    vllm_process = start_vllm_server()
    wait_for_server(PORT)
    log("Initialization complete. Starting RunPod serverless loop...")
except Exception as e:
    log(f"Fatal error during initialization: {e}")


def handler(job):
    job_id = job.id if hasattr(job, 'id') else (job.get('id') if isinstance(job, dict) else 'unknown')
    log(f"Received job: {job_id}")
    job_input = job.input if hasattr(job, 'input') else (job.get('input', {}) if isinstance(job, dict) else {})
    endpoint = job_input.get("endpoint", "chat/completions")
    payload = job_input.get("payload", {})
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]
    local_url = f"http://localhost:{PORT}/v1/{endpoint}"
    log(f"Forwarding job to local endpoint: {local_url}")
    try:
        response = requests.post(local_url, json=payload, timeout=300)
        log(f"Response status from local vLLM: {response.status_code}")
        return {
            "status_code": response.status_code,
            "data": response.json()
        }
    except Exception as e:
        log(f"Error forwarding job: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
