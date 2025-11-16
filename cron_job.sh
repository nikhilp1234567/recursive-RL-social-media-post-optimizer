#!/bin/bash

# Daily Reinforcement Learning Loop Runner
# This script is called by cron to run the RL training and posting workflow

# to add this to the vm run crontab -e and add the following line:
# 0 9 * * * /home/azureuser/RL_AI/cron_job.sh >> /home/azureuser/RL_AI/logs/cron_$(date +\%Y-\%m-\%d).log 2>&1

# Set the project directory
# PROJECT_DIR="/home/azureuser/RL_AI"
PROJECT_DIR="."
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

# Run contextual prompt model first (generates data for fine_tuning_model.py)
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting contextual prompt model..." >> "logs/rl_loop_$(date +%Y-%m-%d).log"
python3 contextual_prompt_model.py >> "logs/rl_loop_$(date +%Y-%m-%d).log" 2>&1

# Capture exit code from contextual prompt model
CONTEXTUAL_EXIT_CODE=$?

# Log completion status but continue regardless
if [ $CONTEXTUAL_EXIT_CODE -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Contextual prompt model failed with exit code $CONTEXTUAL_EXIT_CODE. Continuing with fine-tuning using previous data." >> "logs/rl_loop_$(date +%Y-%m-%d).log"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Contextual prompt model completed successfully. Starting fine-tuning..." >> "logs/rl_loop_$(date +%Y-%m-%d).log"
fi

# Run the reinforcement learning loop
# Redirect output to daily log file with timestamp
python3 fine_tuning_model.py >> "logs/rl_loop_$(date +%Y-%m-%d).log" 2>&1

# Capture exit code
EXIT_CODE=$?

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - RL Loop completed successfully" >> "logs/cron_$(date +%Y-%m-%d).log"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - RL Loop failed with exit code $EXIT_CODE" >> "logs/cron_$(date +%Y-%m-%d).log"
fi

exit $EXIT_CODE