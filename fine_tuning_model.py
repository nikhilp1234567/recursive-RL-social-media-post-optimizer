import json
from helpers.GRPO_Runpod import train_and_generate_post
from helpers.quality_control import clean_generated_post
from helpers.dataset_helpers import add_to_dataset

# Load a prompt from the prompts file
with open('data_africfood/prompts.jsonl', 'r') as f:
    prompt_data = json.loads(f.readlines()[-1])
    custom_prompt = prompt_data['prompt']

# Train the model and generate a post
print("Starting model training and post generation...")
print(f"\nUsing prompt:\n{custom_prompt[:200]}...\n")
print("="*60)
response = train_and_generate_post(
    dataset_path='data_africfood/dataset.jsonl',
    custom_prompt=custom_prompt,
    use_reward_model=True
)

# Clean the cleaned post
post = clean_generated_post(response)

# Display the cleaned post
print("\n" + "="*60)
print("CLEANED TWITTER POST")
print("="*60)
print(post)
print("="*60)

# Add latest post to dataset
add_to_dataset(custom_prompt, post)
