import os
import zarr
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    filename="preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
HISTORY_FRAMES = 10
FUTURE_FRAMES = 20
TOTAL_FRAMES = HISTORY_FRAMES + FUTURE_FRAMES

# Paths
RAW_DATA_PATH = "data/raw/lyft-data/train.zarr"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f"🔍 Opening: {RAW_DATA_PATH}")
zarr_dataset = zarr.open(RAW_DATA_PATH, mode='r')
agents = zarr_dataset['agents']
frames = zarr_dataset['frames']
scenes = zarr_dataset['scenes']
print(f"✅ Zarr dataset loaded: {len(scenes)} scenes")


def extract_scene(scene, scene_idx):
    start_idx, end_idx = scene["frame_index_interval"]
    if (end_idx - start_idx) < TOTAL_FRAMES:
        logging.warning(f"Scene {scene_idx}: too short.")
        return None

    center_idx = start_idx + HISTORY_FRAMES
    frame = frames[center_idx]
    agent_start, agent_end = frame["agent_index_interval"]
    scene_agents = agents[agent_start:agent_end]

    if len(scene_agents) == 0:
        logging.warning(f"Scene {scene_idx}: no agents.")
        return None

    # Use first agent
    agent = scene_agents[0]
    track_id = agent["track_id"]
    trajectory = []

    for f_idx in range(start_idx, start_idx + TOTAL_FRAMES):
        f_agents = agents[frames[f_idx]["agent_index_interval"]
                          [0]: frames[f_idx]["agent_index_interval"][1]]
        match = [a for a in f_agents if a["track_id"] == track_id]
        if match:
            trajectory.append(match[0]["centroid"])
        else:
            logging.warning(
                f"Scene {scene_idx}: missing frame {f_idx} for agent.")
            return None

    trajectory = np.array(trajectory)
    origin = trajectory[HISTORY_FRAMES - 1]
    normed = trajectory - origin
    return normed[:HISTORY_FRAMES], normed[HISTORY_FRAMES:]


# Process scenes
for idx, scene in tqdm(enumerate(scenes), total=len(scenes), desc="⏳ Preprocessing"):
    result = extract_scene(scene, idx)
    if result is None:
        continue

    hist, fut = result
    out_path = os.path.join(PROCESSED_DIR, f"{idx:06}_scene.npz")
    np.savez_compressed(out_path, focal_history=hist, focal_future=fut)

print("✅ Preprocessing completed.")
