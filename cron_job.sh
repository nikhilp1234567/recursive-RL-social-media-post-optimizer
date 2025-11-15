#!/bin/bash

# Daily Reinforcement Learning Loop Runner
# This script is called by cron to run the RL training and posting workflow

# to add this to the vm run crontab -e and add the following line:
# 0 9 * * * /home/azureuser/RL_AI/cron_job.sh >> /home/azureuser/RL_AI/logs/cron_$(date +\%Y-\%m-\%d).log 2>&1

# Set the project directory
PROJECT_DIR="/home/azureuser/RL_AI"
cd "$PROJECT_DIR" || exit 1

# Activate venv
source venv/bin/activate

# Set Python path to project directory
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the reinforcement learning loop
# Redirect output to daily log file with timestamp
python3 reinforcement_learning_loop.py >> "logs/rl_loop_$(date +%Y-%m-%d).log" 2>&1

# Capture exit code
EXIT_CODE=$?

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - RL Loop completed successfully" >> "logs/cron_$(date +%Y-%m-%d).log"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - RL Loop failed with exit code $EXIT_CODE" >> "logs/cron_$(date +%Y-%m-%d).log"
fi

exit $EXIT_CODE