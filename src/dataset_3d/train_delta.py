import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

def pairwise_ranking_loss(pred, target, top_k=8):
    """
    Computes pairwise ranking loss using the target's actual difference as the dynamic margin.
    Restricts the ranking objective to ONLY care about the Top-K most positive and Top-K most negative experts.
    pred: [batch, experts]
    target: [batch, experts]
    """
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    t_diff = target.unsqueeze(2) - target.unsqueeze(1)
    
    B, E = target.shape
    k = min(top_k, E // 2)
    
    # Identify the truly important experts (highest active shift, highest deactivated shift)
    _, top_indices = torch.topk(target, k, dim=-1)
    _, bottom_indices = torch.topk(target, k, dim=-1, largest=False)
    
    important_mask = torch.zeros_like(target, dtype=torch.bool)
    important_mask.scatter_(1, top_indices, True)
    important_mask.scatter_(1, bottom_indices, True)
    
    # Pair is evaluated if AT LEAST ONE of the experts is important
    pair_mask = important_mask.unsqueeze(2) | important_mask.unsqueeze(1)
    
    mask = (t_diff > 1e-5) & pair_mask
    
    # Enforce that predicted rank difference is at least exactly the target difference
    loss_tensor = torch.relu(t_diff - p_diff)
    
    valid_losses = loss_tensor[mask]
    if valid_losses.numel() > 0:
        return valid_losses.mean()
    else:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

def pairwise_ranking_accuracy(pred, target, top_k=8):
    """
    Computes accuracy of the pairwise rankings, focused only on Top-K & Bottom-K.
    """
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    t_diff = target.unsqueeze(2) - target.unsqueeze(1)
    
    B, E = target.shape
    k = min(top_k, E // 2)
    
    _, top_indices = torch.topk(target, k, dim=-1)
    _, bottom_indices = torch.topk(target, k, dim=-1, largest=False)
    
    important_mask = torch.zeros_like(target, dtype=torch.bool)
    important_mask.scatter_(1, top_indices, True)
    important_mask.scatter_(1, bottom_indices, True)
    
    pair_mask = important_mask.unsqueeze(2) | important_mask.unsqueeze(1)
    
    mask = (t_diff > 1e-5) & pair_mask
    correct = (p_diff[mask] > 0).float()
    if correct.numel() > 0:
        return correct.mean().item()
    return 0.0

def topk_intersection(pred, target, k=5):
    """
    Computes the fraction of overlap in the Top-K experts between prediction and target.
    """
    _, pred_topk = torch.topk(pred, k, dim=-1)
    _, target_topk = torch.topk(target, k, dim=-1)
    
    intersections = 0.0
    for p, t in zip(pred_topk, target_topk):
        intersections += len(set(p.tolist()).intersection(set(t.tolist()))) / k
        
    return intersections / pred.shape[0]

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
    
    # Learnable Parameter DELTA [L, D_dim, E]
    # THIS MATCHES "x1.DELTA[l] = e" correctly.
    DELTA = nn.Parameter(torch.randn(L, D_dim, E, device=device) * 0.01)
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
        epoch_top5 = 0.0
        batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_D, batch_target in pbar:
            batch_D = batch_D.to(device)
            batch_target = batch_target.to(device)
            
            # Predict
            # batch_D: [B, L, D_dim]
            # DELTA: [L, D_dim, E]
            # Returns: [B, L, E]
            pred = torch.einsum('bld, lde -> ble', batch_D, DELTA)
            
            pred_flat = pred.reshape(-1, E)
            target_flat = batch_target.reshape(-1, E)
            
            # Combine Pairwise Ranking and Mean Squared Error 
            # to guarantee the scale aligns while preserving extreme ranks of top/bottom K
            loss_rank = pairwise_ranking_loss(pred_flat, target_flat, top_k=args.top_k)
            loss_mse = F.mse_loss(pred_flat, target_flat)
            loss = loss_rank + loss_mse
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            acc = pairwise_ranking_accuracy(pred_flat, target_flat, top_k=args.top_k)
            top5_acc = topk_intersection(pred_flat, target_flat, k=5)
            
            epoch_acc += acc
            epoch_top5 += top5_acc
            batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.4f}", 'top5': f"{top5_acc:.4f}"})
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "batch_loss": loss.item(), 
                    "batch_rank_loss": loss_rank.item(), 
                    "batch_mse_loss": loss_mse.item(), 
                    "batch_acc": acc,
                    "batch_top5_acc": top5_acc
                })
            
        epoch_mean_loss = epoch_loss / batches
        epoch_mean_acc = epoch_acc / batches
        epoch_mean_top5 = epoch_top5 / batches
        print(f"Epoch {epoch+1} | Mean Loss: {epoch_mean_loss:.4f} | Mean Pairwise Acc: {epoch_mean_acc:.4f} | Top-5 Overlap: {epoch_mean_top5:.4f}")
        
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": epoch_mean_loss,
                "epoch_acc": epoch_mean_acc,
                "epoch_top5_acc": epoch_mean_top5
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
    parser.add_argument("--top-k", type=int, default=8, help="Number of Top/Bottom experts to rank. Default 8.")
    parser.add_argument("--margin", type=float, default=0.05, help="Legacy margin arg")
    args = parser.parse_args()
    train(args)
