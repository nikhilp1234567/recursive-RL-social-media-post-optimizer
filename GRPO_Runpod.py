import torch
import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def train_and_generate_post(
    model_name="MoonshotAI/Kimi-K2-Instruct",
    dataset_path="data.jsonl",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    custom_prompt=None
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
    
    Returns:
        str: Generated response from the fine-tuned model
    """
    
    # 1. Load the pre-trained model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # 2. Prepare the model for fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "mlp.o_proj", "mlp.gate_proj", "mlp.up_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # 3. Load and format your dataset
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    except FileNotFoundError:
        print(f"Error: The dataset file '{dataset_path}' was not found.")
        print("Please make sure the file is in the same directory as this script.")
        return None

    def formatting_prompts_func(examples):
        prompts = examples["prompt"]
        posts = examples["post"]
        texts = []
        for prompt, post in zip(prompts, posts):
            texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n{post}{tokenizer.eos_token}")
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Set up and run the SFTTrainer
    trainer = SFTTrainer(
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

    # 5. Start the training process!
    print("Starting fine-tuning...")
    trainer.train()

    # 6. Save the fine-tuned model (LoRA adapters)
    date_str = datetime.date.today().isoformat()
    save_dir = f"lora_model_{date_str}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Fine-tuning complete. Model saved to 'lora_model' directory.")

    # 7. Run inference with the fine-tuned model
    # Load the base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Load the LoRA adapters from the saved directory
    model = FastLanguageModel.get_peft_model(
        model,
        save_dir,
    )
    
    # Generate response
    if custom_prompt is None:
        custom_prompt = input('Enter your prompt for the model: ')
    
    alpaca_prompt = f"### Instruction:\n{custom_prompt}\n\n### Response:\n"
    
    inputs = tokenizer(
        [alpaca_prompt], return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    generated_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print("\n\nGenerated Response:")
    print(generated_response)
    
    return generated_response

# Example usage:
if __name__ == "__main__":
    # Train and generate with default settings
    response = train_and_generate_post()
    
    # Or with custom parameters:
    # response = train_and_generate_post(
    #     dataset_path="my_data.json",
    #     custom_prompt="Write a poem about AI"
    # )
