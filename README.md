# Reinforcement Learning AI Post Generator

This project implements a reinforcement learning system that generates engaging social media posts using GRPO (Generalized Reward Policy Optimization) with a custom reward model.

## Key Features

- **Custom Reward Model**: Trains a neural network to predict post engagement based on historical data
- **GRPO Training**: Uses the reward model to optimize post generation through reinforcement learning
- **Automatic Dataset Management**: Automatically adds new posts to the training dataset
- **Interactive Workflow**: Easy-to-use interface for generating posts and updating metrics

## Files Overview

### Core Components

- `GRPO_Runpod.py`: Main training script that implements GRPO with reward model integration
- `reward_model.py`: Custom reward model implementation using PyTorch and Transformers
- `reinforcement_learning_loop.py`: Interactive workflow manager
- `data.jsonl`: Training dataset in JSONL format (one JSON object per line)

### Supporting Files

- `demo.py`: Demonstration script showing different usage patterns
- `requirements.txt`: Python dependencies
- `scrapers-playwright/`: Web scraping tools for data collection

## Data Format

The dataset (`data.jsonl`) uses this format:

```json
{ "prompt": "Produce an engaging post for twitter", "post": "Your post content here", "views": 42000, "likes": 500, "reposts": 45 }
```

## Workflow

1. **Train Reward Model**: The system trains a neural network on historical posts and their engagement metrics
2. **GRPO Training**: Uses the reward model to guide the language model training with reinforcement learning
3. **Generate New Post**: Creates a new post using the fine-tuned model
4. **Add to Dataset**: Automatically appends the new post to the training data
5. **Update Metrics**: After posting on social media, update the engagement metrics
6. **Repeat**: Run again to continue improving the model

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive workflow
python reinforcement_learning_loop.py
```

### Options

1. **Generate new post**: Runs the full RL workflow (reward model + GRPO training + generation)
2. **Update metrics**: Update engagement metrics for existing posts
3. **Demo mode**: Run `python demo.py` for guided demonstrations

### Programmatic Usage

```python
from GRPO_Runpod import train_and_generate_post
from reward_model import RewardModelTrainer

# Generate a post with reward model
response = train_and_generate_post(
    dataset_path="data.jsonl",
    custom_prompt="Write an engaging tech post",
    use_reward_model=True
)
```

## How the Reward Model Works

1. **Architecture**: Uses a pre-trained language model (ModernBERT) with a regression head
2. **Training Data**: Historical posts with engagement metrics (views + 2×likes + 3×reposts)
3. **Output**: Scalar reward score predicting post engagement
4. **Integration**: Guides GRPO training to generate higher-reward posts

## Key Improvements

- **Learned Rewards**: Replaces simple similarity-based rewards with a trained model
- **Automatic Dataset Growth**: Each generation cycle adds to the training data
- **Better Post Extraction**: Properly extracts generated content from model outputs
- **Interactive Management**: Easy metric updates and workflow control

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth (for efficient training)
- TRL (Transformer Reinforcement Learning)
- Other dependencies in `requirements.txt`

## Model Saving

- Reward models are saved as `reward_model_YYYY-MM-DD/reward_model.pth`
- GRPO models are saved as `lora_model_YYYY-MM-DD/`

## Tips

1. Start with at least 5-10 posts in your dataset for better reward model training
2. Update metrics regularly to improve the reward model
3. Use descriptive prompts for better generation control
4. Monitor the reward scores to ensure the model is learning effectively

## Troubleshooting

- **CUDA errors**: Ensure you have compatible PyTorch and CUDA versions
- **Memory issues**: Reduce batch sizes in the training configuration
- **Generation quality**: Increase training epochs or improve your dataset quality

to
Remote-SSH: Connect to Host

set up the pod

run it

ge ssh command and input into the remote-ssh thing
