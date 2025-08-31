import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Dict, Tuple
import datetime
import os

class RewardModel(nn.Module):
    """
    1. Use ModernBERT to encode post into a vector
    2. Use regression model to take vector input and output scalar reward
    """
    
    def __init__(self, model_name="answerdotai/ModernBERT-base", hidden_size=None, freeze_encoder=True):
        super(RewardModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze the encoder for small datasets
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("ModernBERT encoder frozen: only train regression head")
        else:
            print("ModernBERT encoder trainable: full parameter SFT")
        
        # Get the actual hidden size from the model configuration
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size
        
        self.hidden_size = hidden_size
        self.freeze_encoder = freeze_encoder
        
        # Add padding token if it doesn't exist (ModernBERT has this, Qwen does not)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Regression head that is trained
        # I am using a neural network now, might replace with random forest
        # if neural network is underfitting
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation for BERT
        pooled_output = outputs.last_hidden_state[:, 0, :]

        reward = self.reward_head(pooled_output)
        return reward.squeeze(-1)


class RewardDataset(Dataset):
    # create dataset object for training using pytorch
    def __init__(self, posts: List[str], rewards: List[float], tokenizer, max_length=512):
        self.posts = posts
        self.rewards = rewards
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, idx):
        post = str(self.posts[idx])
        reward = float(self.rewards[idx])
        
        encoding = self.tokenizer(
            post,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'reward': torch.tensor(reward, dtype=torch.float32)
        }


class RewardModelTrainer:
    def __init__(self, model_name="answerdotai/ModernBERT-base", freeze_encoder=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RewardModel(model_name, freeze_encoder=freeze_encoder).to(self.device)
        self.tokenizer = self.model.tokenizer
        
    def prepare_data(self, dataset_path: str) -> Tuple[List[str], List[float]]:
        posts = []
        rewards = []
        
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                post = data['post']

                # A reward that combines views, likes, reposts
                reward = data['views'] + (2 * data['likes']) + (3 * data['reposts'])
            
                posts.append(post)
                rewards.append(reward)
        
        return posts, rewards
    
    def train(self, dataset_path: str, epochs=10, batch_size=8, learning_rate=2e-5):
        posts, rewards = self.prepare_data(dataset_path)
        
        # Store normalization parameters for denormalization during inference
        rewards = np.array(rewards)
        self.reward_min = float(rewards.min())
        self.reward_max = float(rewards.max())
        self.reward_range = self.reward_max - self.reward_min + 1e-8
        
        # Normalize rewards to [0, 1] range for stable neural network training
        rewards_normalized = (rewards - self.reward_min) / self.reward_range
        
        # train test split
        train_posts, val_posts, train_rewards, val_rewards = train_test_split(
            posts, rewards_normalized, test_size=0.2, random_state=42
        )
        
        train_dataset = RewardDataset(train_posts, train_rewards, self.tokenizer)
        val_dataset = RewardDataset(val_posts, val_rewards, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        

        if self.model.freeze_encoder:
            # Only train the regression head parameters
            optimizer = torch.optim.AdamW(self.model.reward_head.parameters(), lr=learning_rate)
        else:
            # Train all parameters with different learning rates
            optimizer = torch.optim.AdamW([
                {'params': self.model.backbone.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for backbone
                {'params': self.model.reward_head.parameters(), 'lr': learning_rate}        # Full LR for head
            ])
            print("Training all parameters with differential learning rates")
        
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # training set
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                rewards = batch['reward'].to(self.device)
                
                optimizer.zero_grad()
                
                predicted_rewards = self.model(input_ids, attention_mask)
                loss = criterion(predicted_rewards, rewards)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # validation set
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    rewards = batch['reward'].to(self.device)
                    
                    predicted_rewards = self.model(input_ids, attention_mask)
                    loss = criterion(predicted_rewards, rewards)
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}\nTrain Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n\n")
    
    def predict_reward(self, text: str, denormalize=True) -> float:
        """takes in a post, outputs a predicted reward"""
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            normalized_reward = self.model(input_ids, attention_mask)
        
        if denormalize and hasattr(self, 'reward_min') and hasattr(self, 'reward_max'):
            reward = normalized_reward.item() * self.reward_range + self.reward_min
            return reward
        else:
            # if we decide to use normalised rewards instead 
            # (not convinced this makes any difference for GRPO)
            return normalized_reward.item()
    
    def save_model(self, save_dir: str = None):
        if save_dir is None:
            date_str = datetime.date.today().isoformat()
            save_dir = f"reward_model_{date_str}"
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'reward_min': getattr(self, 'reward_min', 0),
            'reward_max': getattr(self, 'reward_max', 1),
            'reward_range': getattr(self, 'reward_range', 1)
        }, f"{save_dir}/reward_model.pth")
        
        print(f"Reward model saved to {save_dir}/reward_model.pth")
        print(f"Normalization parameters: min={getattr(self, 'reward_min', 0):.2f}, max={getattr(self, 'reward_max', 1):.2f}")
        return save_dir
    
    def load_model(self, model_path: str):
        """Load a pretrained reward model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalization parameters
        self.reward_min = checkpoint.get('reward_min', 0)
        self.reward_max = checkpoint.get('reward_max', 1)
        self.reward_range = checkpoint.get('reward_range', 1)
        
        print(f"Reward model loaded from {model_path}")
        print(f"Reward range: [{self.reward_min:.2f}, {self.reward_max:.2f}]")
    
    def unfreeze_encoder(self):
        """Unfreeze the ModernBERT encoder for full parameter SFT."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        self.model.freeze_encoder = False
    
    def freeze_encoder(self):
        """Freeze the ModernBERT encoder to only train regression head.
        I am setting this as default due to small training dataset"""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.model.freeze_encoder = True


def create_grpo_reward_function(reward_model_trainer):
    """Create a wrapper reward function for GRPO."""
    
    def reward_function(completions, **kwargs):
        rewards = []
        
        for completion in completions:
            text = completion[0]["content"]
            
            reward = reward_model_trainer.predict_reward(text, denormalize=True)
            rewards.append(reward)
        
        return rewards
    
    return reward_function
