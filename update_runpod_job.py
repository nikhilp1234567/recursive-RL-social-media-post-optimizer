from helpers.cron_job_helpers import terminate_pod, execute_command, create_pod, wait_for_pod_running


if __name__ == "__main__":
    pod_id = None
    try:
        pod_id = create_pod()
        
        if pod_id:
            if wait_for_pod_running(pod_id):
                execute_command(pod_id, "git pull")
    except Exception as e:
        print(f"An unexpected error occurred during the workflow: {e}")
    finally:
        if pod_id:
            terminate_pod(pod_id)