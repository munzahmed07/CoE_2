# src/contrastive/visualize_embeddings.py
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
import random

# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contrastive.encoder_model import TrajectoryEncoder
from core.data_loader import LyftTrajectoryDataset

# --- Configuration ---
ENCODER_PATH = "models/encoder/encoder.pth"
NUM_SAMPLES = 500 # How many trajectories to plot
DEVICE = "cpu" # t-SNE runs on CPU
SAVE_PATH = "results/plots/tsne_embedding_visualization.png"

def label_trajectory(traj):
    """A simple function to label a trajectory as 'straight', 'left', or 'right'."""
    start_point = traj[0]
    end_point = traj[-1]
    # Check the change in x and y
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    if abs(dx) < 0.5: # Mostly vertical movement
        return 'straight'
    if abs(dy) < 0.5: # Mostly horizontal movement
        return 'straight'
    # Simple check for turns based on the ratio of change
    if dx < 0 and dy > 0:
        return 'left_turn'
    if dx > 0 and dy < 0:
        return 'right_turn'
    return 'straight'

if __name__ == "__main__":
    print("🚀 Creating t-SNE visualization for Phase 1 Encoder...")
    os.makedirs("results/plots", exist_ok=True)

    # 1. Load the pre-trained encoder model
    model = TrajectoryEncoder().to(DEVICE)
    model.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Encoder model loaded.")

    # 2. Load a random subset of data
    dataset = LyftTrajectoryDataset(processed_dir="data/processed")
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    
    embeddings = []
    labels = []
    with torch.no_grad():
        for i in indices:
            sample = dataset[i]
            history = sample["focal_history"].unsqueeze(0).to(DEVICE)
            
            # Get the 64-dimensional embedding
            emb = model(history).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            
            # Get a simple label for visualization
            labels.append(label_trajectory(history.squeeze(0).cpu().numpy()))
    
    embeddings = np.array(embeddings)
    print(f"✅ Generated {len(embeddings)} embeddings.")

    # 3. Use t-SNE to reduce embeddings from 64D to 2D
    print("🧠 Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    print("✅ t-SNE complete.")

    # 4. Create a scatter plot
    plt.figure(figsize=(12, 10))
    colors = {'straight': 'blue', 'left_turn': 'green', 'right_turn': 'red'}
    
    for label_name in set(labels):
        # Find all points that have this label
        points = embeddings_2d[np.array(labels) == label_name]
        plt.scatter(points[:, 0], points[:, 1], c=colors.get(label_name, 'gray'), label=label_name)

    plt.title('t-SNE Visualization of Trajectory Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_PATH)
    print(f"✅ t-SNE plot saved to {SAVE_PATH}")