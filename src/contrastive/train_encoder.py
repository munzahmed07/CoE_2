import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

from encoder_model import TrajectoryEncoder
from loss import info_nce_loss
# We assume the core data_loader is one level up
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_loader import LyftTrajectoryDataset 

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 64
NUM_NEGATIVES = 5 # Number of negative samples per anchor

class ContrastiveDataset(Dataset):
    """
    A dataset that generates anchor, positive, and negative trajectory pairs on-the-fly.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Anchor trajectory is the one at the current index
        anchor_data = self.base_dataset[idx]
        anchor_traj = anchor_data["focal_history"]

        # The key fix is here:
        # A "positive" sample is a slightly augmented version of the anchor.
        # This ensures the dimensions and length are always identical.
        positive_traj = anchor_traj + torch.randn_like(anchor_traj) * 0.01 # Add tiny noise

        # Negative trajectories are from random other agents
        negative_indices = random.sample(range(len(self.base_dataset)), NUM_NEGATIVES)
        negative_trajs = [self.base_dataset[i]["focal_history"] for i in negative_indices]

        return anchor_traj, positive_traj, torch.stack(negative_trajs)


if __name__ == "__main__":
    print("🚀 Starting Phase 1: Contrastive Encoder Training")
    
    # 1. Load data and create contrastive dataset
    base_dataset = LyftTrajectoryDataset(processed_dir="data/processed")
    contrastive_dataset = ContrastiveDataset(base_dataset)
    train_loader = DataLoader(contrastive_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Data loaded. Found {len(base_dataset)} samples.")

    # 2. Initialize model and optimizer
    model = TrajectoryEncoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"✅ Model and optimizer initialized on {DEVICE}.")

    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for anchor, positive, negatives in train_loader:
            anchor, positive, negatives = anchor.to(DEVICE), positive.to(DEVICE), negatives.to(DEVICE)
            
            # Get embeddings
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            
            # Reshape negatives for batch processing
            b, n, s, f = negatives.shape
            negative_embs = model(negatives.view(-1, s, f)).view(b, n, -1)

            # Calculate loss
            loss = info_nce_loss(anchor_emb, positive_emb, negative_embs)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    # 4. Save the trained model
    os.makedirs("models/encoder", exist_ok=True)
    torch.save(model.state_dict(), "models/encoder/encoder.pth")
    print("✅ Encoder model saved to models/encoder/encoder.pth")
    print("🎉 Phase 1 Complete!")