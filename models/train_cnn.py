# models/train_cnn.py
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


class ScreenshotDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.meta = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        # Binary label: 1 = poor usability (score >= 0.4), 0 = good
        score = float(row["usability_score"])
        label = 1 if score >= 0.4 else 0

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def train(csv_path, epochs=5, batch_size=8):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    # Full dataset
    full_ds = ScreenshotDataset(csv_path, transform)

    # Train/validation split
    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_total, train_correct = 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = crit(out, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += labels.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ---- Validate ----
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_total += labels.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), "models/cnn_model.pth")
    print("✅ CNN model saved at models/cnn_model.pth")


if __name__ == "__main__":
    train("data/labelled/metadata.csv")
