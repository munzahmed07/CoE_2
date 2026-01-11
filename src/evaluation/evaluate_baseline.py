import torch
from torch.utils.data import DataLoader
from src.data_loader import LyftTrajectoryDataset
from src.model import LSTMPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(pred, gt):
    ade = torch.mean(torch.norm(pred - gt, dim=2))
    fde = torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=1))
    return ade.item(), fde.item()


def evaluate(model, val_loader):
    model.eval()
    total_ade = 0
    total_fde = 0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            history = batch["focal_history"].to(DEVICE)
            future = batch["focal_future"].to(DEVICE)

            pred = model(history)

            ade, fde = compute_metrics(pred, future)
            total_ade += ade * history.size(0)
            total_fde += fde * history.size(0)
            count += history.size(0)

    print(f"\n📈 Evaluation Results:")
    print(f"   🔹 ADE: {total_ade / count:.4f}")
    print(f"   🔹 FDE: {total_fde / count:.4f}")



if __name__ == "__main__":
    print("🔍 Loading validation dataset...")
    dataset = LyftTrajectoryDataset("data/processed")
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("📦 Loading model...")
    model = LSTMPredictor().to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE, weights_only=True))

    evaluate(model, val_loader)


