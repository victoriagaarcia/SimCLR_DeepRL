import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PLOTS_DIR = "plots"

def subplot_tsne_2x2(save_name="part4_tsne_2x2.png"):
    # Ajusta estos nombres si tus ficheros se llaman distinto
    files = [
        ("latent_2d_0005_tsne.png", "t-SNE (τ = 0.05)"),
        ("latent_2d_001_tsne.png",  "t-SNE (τ = 0.10)"),
        ("latent_2d_tsne2.png",  "t-SNE (τ = 0.20)"),
        ("latent_2d_005_tsne.png",  "t-SNE (τ = 0.50)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (fname, title) in zip(axes, files):
        path = os.path.join(PLOTS_DIR, fname)
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.suptitle("CIFAR-10 latent space (t-SNE) for different temperatures", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    subplot_tsne_2x2()
