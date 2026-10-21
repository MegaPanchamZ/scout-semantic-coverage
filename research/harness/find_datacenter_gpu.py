import json
import subprocess
import os

# Datacenters that support network volumes (from the error message)
NET_VOL_DCS = {
    "AP-JP-1", "CA-MTL-3", "CA-MTL-4", "EU-CZ-1", "EU-FR-1", "EU-NL-1", "EU-RO-1",
    "EU-SE-1", "EUR-IS-1", "EUR-IS-3", "EUR-NO-1", "EUR-NO-2", "US-CA-2", "US-GA-2",
    "US-IL-1", "US-KS-2", "US-MO-1", "US-MO-2", "US-NC-1", "US-NC-2", "US-NE-1",
    "US-TX-3", "US-WA-1"
}

def main():
    print("Running runpodctl to list datacenters...")
    result = subprocess.run(["runpodctl", "datacenter", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to run runpodctl:", result.stderr)
        return
        
    try:
        data = json.loads(result.stdout)
    except Exception as e:
        print("Failed to parse JSON:", e)
        # Try finding json bracket
        idx = result.stdout.find("[")
        if idx != -1:
            try:
                data = json.loads(result.stdout[idx:])
            except Exception as e2:
                print("Fallback JSON parsing failed:", e2)
                return
        else:
            return

    # Filter to datacenters supporting network volumes
    print("\nGPU Availability in Net-Volume-Supported Datacenters:")
    found_any = False
    for dc in data:
        dc_id = dc.get("id")
        if dc_id in NET_VOL_DCS:
            gpus = dc.get("gpuAvailability", [])
            in_stock_gpus = [g for g in gpus if g.get("stockStatus") in ("Low", "Medium", "High")]
            if in_stock_gpus:
                found_any = True
                print(f"\nDatacenter: {dc_id} ({dc.get('name')})")
                for g in in_stock_gpus:
                    print(f"  - {g.get('displayName')} ({g.get('gpuId')}): Stock = {g.get('stockStatus')}")
                    
    if not found_any:
        print("No GPUs in stock in any net-volume-supported datacenters.")

if __name__ == "__main__":
    main()
