import os
import zarr

# Adjust the absolute or relative path based on your directory
path = os.path.join("data", "raw", "lyft-data", "train.zarr")

print(f"🔍 Trying to open: {os.path.abspath(path)}")

try:
    root = zarr.open(path, mode='r')
    print("✅ Zarr dataset opened.")
    print(f"  ➤ Scenes: {len(root['scenes'])}")
    print(f"  ➤ Frames: {len(root['frames'])}")
    print(f"  ➤ Agents: {len(root['agents'])}")
except Exception as e:
    print(f"❌ Failed to open Zarr dataset: {e}")
