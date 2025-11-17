import torch, os, datetime, time, numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from transformers import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from helpers.model_reward import RewardModelTrainer, create_grpo_reward_function
from helpers.paths import DATASET_FILE

def train_and_generate_post(
    model_name="unsloth/Mistral-7B-Instruct-v0.3",
    dataset_path=DATASET_FILE,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    custom_prompt=None,
    use_reward_model=True
):
    """
    Fine-tune a language model and generate a response to a prompt.
    
    Args:
        model_name (str): Hugging Face model name
        dataset_path (str): Path to JSON dataset file
        max_seq_length (int): Maximum sequence length for the model
        dtype: Data type for the model (None for auto-detection)
        load_in_4bit (bool): Whether to load model in 4-bit
        custom_prompt (str): Custom prompt for generation (if None, will prompt user)
        use_reward_model (bool): Whether to use trained reward model or dummy function
    
    Returns:
        str: Generated response from the fine-tuned model
    """
    
    # Initialize timing
    start_time = time.time()
    stage_times = {}
    
    def log_stage_time(stage_name, stage_start):
        stage_duration = time.time() - stage_start
        stage_times[stage_name] = stage_duration
        print(f"⏱️  {stage_name} completed in {stage_duration:.2f} seconds")
        return time.time()

    # === 1. Load the pre-trained model with Unsloth ===
    # This loads a HuggingFace model and tokenizer using Unsloth's FastLanguageModel,
    # which is optimized for fast fine-tuning and inference.
    print("\n=== 1. Load the pre-trained model with Unsloth ===\n")
    # if os.path.exists("lora_model"):
    #     model_name = "lora_model"
    #     print("Loading model from 'lora_model' directory")

    stage_start = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    # stage_start = log_stage_time("Model Loading", stage_start)

    # === 2. Prepare the model for parameter-efficient fine-tuning (PEFT) ===
    # This wraps the model with LoRA adapters, which allow efficient fine-tuning
    # by only training a small number of additional parameters.
    print("\n=== 2. Prepare the model for PEFT with LoRA adapters ===\n")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "mlp.o_proj", "mlp.gate_proj", "mlp.up_proj",
        ],  # Which modules to apply LoRA to
        lora_alpha=16,  # LoRA scaling factor
        lora_dropout=0,  # No dropout for LoRA
        bias="none",  # No bias adaptation
        use_gradient_checkpointing="unsloth",  # Save memory during training
        random_state=3407,  # For reproducibility
        use_rslora=False,  # Don't use random sign LoRA
    )
    # stage_start = log_stage_time("PEFT Model Preparation", stage_start)

    # === 3. Load and format your dataset ===
    # Try to load the dataset from a JSONL file using HuggingFace Datasets.
    print("\n=== 3. Load and format dataset ===\n")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    except FileNotFoundError:
        print(f"Error: The dataset file '{dataset_path}' was not found.")
        print("Please make sure the file is in the same directory as this script.")
        return None

    # Helper function to format data for SFT (Supervised Fine-Tuning)
    print("Setting formatting functions for the dataset")
    def formatting_prompts_func(examples):
        prompts = examples["prompt"]
        posts = examples["post"]
        texts = []
        for prompt, post in zip(prompts, posts):
            # Format as instruction-following prompt/response
            texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n{post}{tokenizer.eos_token}")
        return {"text": texts}
    
    # Helper function to format data for GRPO (Reinforcement Learning)
    def formatting_prompt_grpo(example):
        return {
            "prompt": [
                {"role": "user", "content": f"### Instruction:\n{example['prompt']}\n\n### Response:"}
            ],
            "answer": example["post"]
        }
    
    # Apply formatting to the dataset for both SFT and GRPO
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    formatted_dataset_grpo = dataset.map(formatting_prompt_grpo)
    stage_start = log_stage_time("Dataset Loading and Formatting", stage_start)

    # === 4. Set up reward function for GRPO ===
    print("\n=== 4. Train reward model for GRPO ===\n")
    if use_reward_model:
        # If using a learned reward model, train it on the dataset
        print("Training reward model...")
        reward_trainer = RewardModelTrainer(freeze_encoder=True)
        reward_trainer.train(dataset_path, epochs=5)
        reward_trainer.save_model()
        
        # Create a reward function that uses the trained reward model
        reward_function = create_grpo_reward_function(reward_trainer)
        print("Reward model training complete. Using trained reward model for GRPO.")
    else:
        # If not using a reward model, use a dummy reward function based on BERT similarity and post metrics
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def calculate_bert_similarity(text1, text2):
            """Calculate cosine similarity between two texts using BERT embeddings."""
            embeddings = bert_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        
        def reward_function_dummy(completions, answer, **kwargs):
            # Calculate a reward for each completion based on similarity and post metrics
            rewards = []
            post_reward = kwargs["views"] + (2 * kwargs["likes"]) + (3 * kwargs["reposts"])
            for completion in completions:
                comp = completion[0]["content"]

                # get BERT similarity score between comp and answer
                similarity_score = calculate_bert_similarity(comp, answer)
                rewards.append(similarity_score * post_reward)

            return rewards
        
        reward_function = reward_function_dummy
        print("Using dummy reward function for GRPO.")
    
    # stage_start = log_stage_time("Reward Function Setup", stage_start)
    
    # === 5. Set up and run the trainers ===
    print("\n=== 5. Set up GRPO trainers ===\n")
    # SFTTrainer: Supervised fine-tuning (not used for training here, but can be used for comparison)
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
    )

    # GRPOTrainer: Reinforcement learning with the reward function
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset_grpo,
        reward_funcs=reward_function,
        args=GRPOConfig(
            num_generations=8,  # How many completions to sample per prompt
            learning_rate=2e-5,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_8bit",
            report_to=None,
        ),
    )
    # stage_start = log_stage_time("Trainer Setup", stage_start)

    # === 6. Start the training process! ===
    # We use the GRPO trainer for RL fine-tuning.
    print("\n=== 6. Start the training process ===\n")
    trainer = grpo_trainer
    
    print("Starting training...")
    trainer.train()
    stage_start = log_stage_time("Model Training", stage_start)

    # === 7. Save the fine-tuned model (LoRA adapters) ===
    # Save the model and tokenizer to a directory named with today's date
    print("\n=== 7. Save the fine-tuned model ===\n")
    date_str = datetime.date.today().isoformat()
    save_dir = f"lora_model_{date_str}"
    # model.save_pretrained(model_name)
    # tokenizer.save_pretrained(model_name)
    model.save_pretrained_merged(
        save_dir,
        tokenizer,
        # save_method = "merged_16bit"
    )
    print(f"Fine-tuning complete. Model saved to '{save_dir}' directory.")
    stage_start = log_stage_time("Model Saving", stage_start)

    # === 8. Run inference with the fine-tuned model ===
    print("\n=== 8. Run inference with the fine-tuned model ===\n")
    stage_start = time.time()
    # Load the updated model for inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name=model_name,
        model_name=save_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    FastLanguageModel.for_inference(model)
    
    # Load the LoRA adapters from the saved directory to apply the fine-tuned weights
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     save_dir,
    # )
    
    
    # If no custom prompt is provided, ask the user for one
    if custom_prompt is None:
        custom_prompt = 'Produce an engaging post for twitter'
    
    # Format the prompt in Alpaca-style instruction format
    # alpaca_prompt = f"### Instruction:\n{custom_prompt}\n\n### Response:\n"
    
    messages = [
    {"role": "system", "content": "Only output the final tweet text. No analysis or labels."},
    {"role": "user", "content": custom_prompt},
]

    alpaca_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt and move tensors to GPU
    inputs = tokenizer(
        [alpaca_prompt], return_tensors="pt"
    ).to("cuda")

    # Generate a response from the model
    outputs = model.generate(**inputs, max_new_tokens=800, use_cache=True)
    generated_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # stage_start = log_stage_time("Text Generation", stage_start)
    
    # Print timing summary
    # total_time = time.time() - start_time
    # print("\n" + "="*60)
    # print("🕒 TIMING SUMMARY")
    # print("="*60)
    # for stage, duration in stage_times.items():
    #     percentage = (duration / total_time) * 100
    #     print(f"{stage:<30}: {duration:>8.2f}s ({percentage:>5.1f}%)")
    # print("-"*60)
    # print(f"{'Total Execution Time':<30}: {total_time:>8.2f}s (100.0%)")
    # print("="*60)

    # stage_start = log_stage_time("Model Reloading for Inference", stage_start)
    return generated_response

# Example usage:
if __name__ == "__main__":
    # Train and generate with default settings
    response = train_and_generate_post()
    print("\n=== Generated Response ===\n")
    print(response)

