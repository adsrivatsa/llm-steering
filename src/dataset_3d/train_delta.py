import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def pairwise_ranking_loss(pred, target, margin=0.05):
    """
    Computes pairwise ranking loss.
    pred: [batch, experts]
    target: [batch, experts]
    """
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    t_diff = target.unsqueeze(2) - target.unsqueeze(1)
    
    mask = t_diff > 1e-5  # threshold to ignore ties
    
    loss_tensor = torch.relu(margin - p_diff)
    
    valid_losses = loss_tensor[mask]
    if valid_losses.numel() > 0:
        return valid_losses.mean()
    else:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

def pairwise_ranking_accuracy(pred, target):
    """
    Computes accuracy of the pairwise rankings.
    """
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    t_diff = target.unsqueeze(2) - target.unsqueeze(1)
    
    mask = t_diff > 1e-5
    correct = (p_diff[mask] > 0).float()
    if correct.numel() > 0:
        return correct.mean().item()
    return 0.0

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dataset from {args.dataset_path}")
    data = torch.load(args.dataset_path, map_location="cpu", weights_only=False)
    
    D = data["D"].float() # [S*L, hidden_dim]
    y1 = data["y1"].float()
    y2 = data["y2"].float()
    meta = data["metadata"]
    
    L = meta["layers"]
    S = meta["num_samples"]
    D_dim = meta["hidden_dim"]
    E = meta["experts"]
    
    print(f"Dataset metadata: Samples={S}, Layers={L}, Hidden={D_dim}, Experts={E}")
    
    D = D.view(S, L, D_dim)
    y1 = y1.view(S, L, E)
    y2 = y2.view(S, L, E)
    
    target = y1 - y2 # [S, L, E]
    
    dataset = TensorDataset(D, target)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Learnable Parameter DELTA [D_dim, L, E]
    DELTA = nn.Parameter(torch.randn(D_dim, L, E, device=device) * 0.01)
    optimizer = optim.Adam([DELTA], lr=args.lr, weight_decay=args.weight_decay)
    
    if _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project="tokenaware-steering-moe",
            entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
            group="train_delta",
            name=f"train-delta-{args.epochs}ep-{args.batch_size}bs",
            config=vars(args)
        )
        wandb.config.update({"samples": S, "layers": L, "hidden_dim": D_dim, "experts": E})
        print("W&B initialized")
    else:
        print("W&B not available or WANDB_API_KEY not set. Skipping logging.")

    print(f"Starting training on {device}...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_D, batch_target in pbar:
            batch_D = batch_D.to(device)
            batch_target = batch_target.to(device)
            
            # Predict
            # batch_D: [B, L, D_dim]
            # DELTA: [D_dim, L, E]
            # Returns: [B, L, E]
            pred = torch.einsum('bld, dle -> ble', batch_D, DELTA)
            
            pred_flat = pred.reshape(-1, E)
            target_flat = batch_target.reshape(-1, E)
            
            loss = pairwise_ranking_loss(pred_flat, target_flat, margin=args.margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            acc = pairwise_ranking_accuracy(pred_flat, target_flat)
            epoch_acc += acc
            batches += 1
            
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.4f}"})
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"batch_loss": loss.item(), "batch_acc": acc})
            
        epoch_mean_loss = epoch_loss / batches
        epoch_mean_acc = epoch_acc / batches
        print(f"Epoch {epoch+1} | Mean Loss: {epoch_mean_loss:.4f} | Mean Pairwise Acc: {epoch_mean_acc:.4f}")
        
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": epoch_mean_loss,
                "epoch_acc": epoch_mean_acc
            })
        
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(DELTA.detach().cpu(), args.output_path)
    print(f"Saved trained 3D DELTA of shape {tuple(DELTA.shape)} to {args.output_path}")

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D DELTA tensor using Rank Regression")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to input dataset.pt")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save delta.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()
    train(args)
