from helpers.cron_job_helpers import terminate_pod, execute_command, create_pod, wait_for_pod_running, BASH_COMMAND_TO_RUN

# --- Main Workflow ---
if __name__ == "__main__":
    pod_id = None
    try:
        # 1. Create the pod
        pod_id = create_pod()
        
        if pod_id:
            # 2. Wait for the pod to be ready
            if wait_for_pod_running(pod_id):
                # 3. Execute the installation and script command
                print('waiting completed')
                execute_command(pod_id, BASH_COMMAND_TO_RUN)

    except Exception as e:
        print(f"An unexpected error occurred during the workflow: {e}")
    
    # finally:
        # 4. Terminate the pod, regardless of whether the command succeeded or failed
        # if pod_id:
            # terminate_pod(pod_id)