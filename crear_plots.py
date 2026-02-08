# crear_plots.py
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PLOTS_DIR = "plots"

# --- Files ---
UNTRAINED = [
    "latent_2d_pca2_init.png",
    "latent_2d_tsne2_init.png",
    "latent_2d_umap2_init.png",
]

TRAINED = [
    "latent_2d_pca2.png",
    "latent_2d_tsne2.png",
    "latent_2d_umap2.png",
]


def make_subplot(image_files, titles, save_path, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, img_name, title in zip(axes, image_files, titles):
        img_path = os.path.join(PLOTS_DIR, img_name)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[OK] Saved {save_path}")


if __name__ == "__main__":

    # --------- Untrained model ----------
    make_subplot(
        image_files=UNTRAINED,
        titles=["PCA", "t-SNE", "UMAP"],
        save_path=os.path.join(PLOTS_DIR, "latent_untrained_all.png"),
        suptitle="CIFAR-10 Latent Space – Untrained Encoder",
    )

    # --------- Trained model ----------
    make_subplot(
        image_files=TRAINED,
        titles=["PCA", "t-SNE", "UMAP"],
        save_path=os.path.join(PLOTS_DIR, "latent_trained_all.png"),
        suptitle="CIFAR-10 Latent Space – SimCLR Trained Encoder",
    )