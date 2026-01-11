import zarr
import numpy as np
import os

# Adjust path to your actual dataset
dataset_path = r"C:\Users\munze\OneDrive\Desktop\CoE_2\lyft-data\train.zarr"

# Load zarr dataset
zarr_dataset = zarr.open(dataset_path, mode='r')

print("✅ Dataset loaded")

# List top-level keys: frames, agents, scenes, tl_faces, etc.
print("Top-level keys:", list(zarr_dataset.array_keys()))

# Try printing the number of agents
agents = zarr_dataset['agents']
print(f"Total agents: {len(agents)}")
print("Example agent:", agents[0])
