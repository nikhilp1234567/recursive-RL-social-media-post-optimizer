import requests
import os
import time
import sys
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(dotenv_path=find_dotenv())
API_KEY = os.getenv("RUNPOD_API_KEY")
BASE_URL = "https://rest.runpod.io/v1"

if not API_KEY:
    print("❌ Error: RUNPOD_API_KEY environment variable not set.")
    sys.exit(1)


# --- API Headers ---
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def create_pod():
    """Sends a request to create a new pod."""
    url = f"{BASE_URL}/pods"
    
    # Payload from your original script to define the pod's configuration
    payload = {
    "gpuCount": 1,
    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],  # From Image 2
    "interruptible": False,  # For On-Demand pricing as requested
    "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",  # Based on "pod VS code image"
    "name": "Rlai_pod", # A logical name for the VS Code pod
    "networkVolumeId": "4f6cxaxs97",  # From Image 1, "Rlai s3" volume
    "volumeInGb": 50,  # From Image 1, size of the network volume
    "containerDiskInGb": 50, # A reasonable default for VS Code, can be adjusted
    "ports": ["8888/http", "22/tcp"], # Standard ports for Jupyter and SSH, as seen in Image 3
    "env": {}, # No specific environment variables were requested
    "volumeMountPath": "/workspace", # Common mount path
    "minVCPUPerGPU": 2, # As seen in Image 2, 2 vCPU per GPU
    "minRAMPerGPU": 8, # As seen in Image 2, 8 GB RAM per GPU
    "dataCenterIds": ["EU-RO-1"], # From Image 1, "Rlai s3" data center
    "dataCenterPriority": "availability",
    }
    
    print("Sending request to create pod...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        pod_data = response.json()
        print(f"Pod creation initiated. Pod ID: {pod_data['id']}")
        print(pod_data)
        return pod_data
    else:
        print(f"❌ Error creating pod: {response.status_code}")
        print(response.json())
        return None

def wait_for_pod_running(pod_id, timeout_seconds=600, poll_seconds=5):
    """Wait until the pod is RUNNING and connection details are available."""
    url = f"{BASE_URL}/pods/{pod_id}"
    start_time = time.time()
    last_state = None

    print("Waiting for pod to be ready (this can take up to 5 minutes sometimes)...")

    while True:
        if time.time() - start_time > timeout_seconds:
            print("❌ Timed out waiting for pod to be ready.")
            return False

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200 or 201:
                pod = response.json()  # NOTE: top-level, not nested under 'pod'
                desired = pod.get("desiredStatus")
                public_ip = (pod.get("publicIp") or "").strip()
                port_mappings = pod.get("portMappings")

                ready = desired == "RUNNING" and bool(public_ip) and bool(port_mappings)
                if ready:
                    print("Pod is RUNNING and connection details are available.")
                    return public_ip, port_mappings

                # Only log when state changes to avoid spam
                state = (desired, bool(public_ip), bool(port_mappings))
                if state != last_state:
                    last_state = state
            else:
                print(f"   Warning: Received status code {response.status_code} while polling.")
        except requests.exceptions.RequestException as e:
            print(f"   An error occurred while polling: {e}")

        time.sleep(poll_seconds)

import subprocess

def execute_command(ip, port_mappings,bash_command):
    """
    Constructs and executes the predefined bash command on the pod via SSH,
    streaming the output in real-time.
    """
    try:
        ssh_port = port_mappings.get('22')
        print(f"🔎 Found SSH port: {ssh_port}")
    except StopIteration:
        print("❌ Error: Could not find the port for SSH")
        return
    
    command_to_execute = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{ip} -p {ssh_port} -i ~/.ssh/id_ed25519 '{bash_command}'"   
    
    print(f"🏃 Executing command on pod...")
    print(f"   CMD: {command_to_execute}")

    # 3. Execute with subprocess.Popen to stream output
    try:
        # Popen is used instead of run() to get real-time output
        process = subprocess.Popen(
            command_to_execute,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Read and print stdout line by line as it comes in
        print("\n--- Remote Script Output (stdout) ---")
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip()) # .strip() removes trailing newlines
        
        # Capture any final error output
        stderr_output = process.stderr.read()
        
        # Check the final return code of the process
        if process.returncode == 0:
            print("\n✅ Command executed successfully.")
        else:
            print(f"\n❌ Command failed with exit code: {process.returncode}")
            if stderr_output:
                print("\n--- Error Output (stderr) ---")
                print(stderr_output.strip())

    except Exception as e:
        print(f"\n❌ An exception occurred while trying to execute the command: {e}")
        
def terminate_pod(pod_id):
    """Sends a request to terminate the pod."""
    url = f"{BASE_URL}/pods/{pod_id}"
    
    print(f"🔥 Terminating pod {pod_id}...")
    response = requests.delete(url, headers=headers)
    
    if response.status_code == 204:
        print("✅ Pod termination initiated successfully.")
    else:
        print(f"❌ Error terminating pod: {response.status_code}")
        print(response.json())
