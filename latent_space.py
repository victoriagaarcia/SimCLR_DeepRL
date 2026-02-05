import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from linear_probe import get_encoder, get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def extract_embeddings(encoder: nn.Module, data_loader: DataLoader):
    encoder.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting Embeddings"):
            images = images.to(device)
            # Extract features using the frozen encoder
            features = encoder(images) # [batch_size, in_dim]
            # Store in CPU memory for numpy compatibility
            embeddings_list.append(features.detach().cpu())
            labels_list.append(labels.cpu())

    embeddings = torch.cat(embeddings_list, dim=0) # [num_samples, in_dim]
    labels = torch.cat(labels_list, dim=0) # [num_samples]
    return embeddings, labels

def reduce_pca(X, seed=42):
    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X)
    return X2, pca

def reduce_tsne(X, seed=42, perplexity=30.0):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    X2 = tsne.fit_transform(X)
    return X2, tsne 

def plot_2d(X2, y, title, save_path, show=False):
    plt.figure(figsize=(10, 8))

    # Colormap discreto con 10 colores
    cmap = plt.get_cmap("tab10")

    for class_idx in range(10):
        mask = (y == class_idx)
        plt.scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=10,
            alpha=0.8,
            color=cmap(class_idx),
            label=CIFAR10_CLASSES[class_idx],
        )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(markerscale=2, fontsize=9, loc="best")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[latent_space] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/Feb02_18_31_27/checkpoint_final_0300.pth.tar")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--data_dir", default="./datasets")
    ap.add_argument("--split", choices=["train", "test"], default="test",
                    help="Which split to visualize")
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca")
    ap.add_argument("--n_samples", type=int, default=5000,
                    help="How many samples to visualize (0 = all). For t-SNE use 2000-5000.")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="plots/latent_2d.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    
    encoder, in_dim = get_encoder(args.ckpt, args.arch, out_dim=args.out_dim)
    print(f"[latent_space] Loaded encoder. in_dim={in_dim}")

    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)
    loader = train_loader if args.split == "train" else test_loader
    
    embeddings, labels = extract_embeddings(encoder, loader)
    X = embeddings.numpy()
    y = labels.numpy()
    
    if args.n_samples and args.n_samples > 0 and args.n_samples < len(y):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y), size=args.n_samples, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"[latent_space] Subsampled to n={len(y)}")
    
    if args.method == "pca":
        X2, _ = reduce_pca(X, seed=args.seed)
    else:
        X2, _ = reduce_tsne(X, seed=args.seed, perplexity=args.perplexity)
    
    title = f"CIFAR-10 latent space ({args.method.upper()}) | {args.split} | arch={args.arch} | n={len(y)}"
    plot_2d(X2, y, title=title, save_path=args.save, show=args.show)
    
if __name__ == "__main__":
    main()