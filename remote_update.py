from helpers.cron_job_helpers import terminate_pod, execute_command, create_pod, wait_for_pod_running

# --- Main Workflow ---
if __name__ == "__main__":
    pod_id = None
    try:
        # 1. Create the pod
        pod_id = create_pod()
        
        # if pod_id:
        #     2. Wait for the pod to be ready
        result = wait_for_pod_running(pod_id['id'])
        if result:
            public_ip, port_mappings = result
            print(f'Pod is ready! IP: {public_ip}, Ports: {port_mappings}')
            command = "cp -a /workspace/.ssh/. /root/.ssh/ && chmod 600 /root/.ssh/id_ed25519 && cd /workspace/RL_AI && git pull"
            execute_command(public_ip, port_mappings, command)

    except Exception as e:
        print(f"An unexpected error occurred during the workflow: {e}")
    
    finally:
        # 4. Terminate the pod, regardless of whether the command succeeded or failed
        if pod_id:
            terminate_pod(pod_id['id'])