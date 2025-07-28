import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LyftTrajectoryDataset(Dataset):
    def __init__(self, processed_dir="data/processed"):
        self.files = sorted([
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir)
            if f.endswith(".npz")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        hist = torch.tensor(data["focal_history"], dtype=torch.float32)
        fut = torch.tensor(data["focal_future"], dtype=torch.float32)

        t_hist = torch.linspace(-10, -1, steps=hist.shape[0]).unsqueeze(1)
        hist = torch.cat([hist, t_hist], dim=1)

        return {"focal_history": hist, "focal_future": fut}

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = LyftTrajectoryDataset("data/processed")
    print("✅ Total:", len(ds))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print("🔹 History:", batch["focal_history"].shape)
    print("🔹 Future :", batch["focal_future"].shape)

