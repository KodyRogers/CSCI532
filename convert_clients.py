import numpy as np
import torch
from PIL import Image
import os
import shutil

# -------------------------
# Partition into clients
# -------------------------
def partition_clients(X, y, num_clients=20):
    num_classes = len(np.unique(y))
    indices_by_class = {i: np.where(y == i)[0] for i in range(num_classes)}

    client_indices = [[] for _ in range(num_clients)]
    for label, indices in indices_by_class.items():
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for cid in range(num_clients):
            client_indices[cid].extend(splits[cid])

    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    return client_indices

# -------------------------
# Save client data
# -------------------------
def save_client_data(cid, X, y, out_dir, num_images=50):
    # Save as .pt (tuple of numpy arrays)
    pt_path = os.path.join(out_dir, f"client_{cid}.pt")
    torch.save((X, y), pt_path)
    print(f"[SAVED] {pt_path}")

    # Save sample PNGs
    img_dir = os.path.join(out_dir, f"client_{cid}_images")
    os.makedirs(img_dir, exist_ok=True)

    for i in range(min(num_images, len(X))):
        img = (X[i] * 255).astype(np.uint8)  # back to [0,255] for saving
        label = y[i]

        # Handle grayscale vs RGB
        if img.shape[-1] == 3:  # HWC RGB
            pil_img = Image.fromarray(img)
        else:  # Grayscale (H, W)
            pil_img = Image.fromarray(img.squeeze(), mode="L")

        pil_img.save(os.path.join(img_dir, f"sample_{i}_label{label}.png"))

    print(f"[EXPORTED] {num_images} images to {img_dir}")

# -------------------------
# Main
# -------------------------
def main():
    OUT_DIR = "clients_data"
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load cleaned PathMNIST
    data = np.load("pathmnist_cleaned.npz")
    X_train, y_train = data["train_images"], data["train_labels"]

    # Partition into 20 clients
    client_indices = partition_clients(X_train, y_train, num_clients=20)

    # Save each client
    for cid, idxs in enumerate(client_indices):
        save_client_data(cid, X_train[idxs], y_train[idxs], OUT_DIR, num_images=50)

    print("[DONE] All client datasets saved in clients_data/")

if __name__ == "__main__":
    main()
