import numpy as np
import torch
from PIL import Image
import os
import shutil

# PathMNIST label map (9 classes)
LABEL_MAP = {
    0: "Adipose tissue",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth muscle",
    6: "Normal colon mucosa",
    7: "Cancer-associated stroma",
    8: "Colorectal adenocarcinoma epithelium"
}

# -------------------------
# Partition aligned across clients
# -------------------------
def partition_clients_aligned(X, y, num_clients=20):
    
    num_classes = len(np.unique(y))
    indices_by_class = {}

    for class_id in range(num_classes):
        class_indices = np.where(y == class_id)[0]
        indices_by_class[class_id] = class_indices

    # Shuffle each class pool
    for lable in indices_by_class:
        np.random.shuffle(indices_by_class[lable])

    # Minimum samples available across all classes
    lengths = []
    for indices in indices_by_class.values():
        lengths.append(len(indices))

    min_per_class = min(lengths)

    # Number of samples per client = min_per_class
    num_samples = min_per_class

    # Build aligned partitions
    client_indices = []

    for _ in range(num_clients):
        client_indices.append([])

    for sample_idx in range(num_samples):
        # Pick a label in round-robin fashion
        lable = sample_idx % num_classes
        # Take one sample for each client from this label pool
        for client_id in range(num_clients):
            pick = indices_by_class[lable][sample_idx]
            client_indices[client_id].append(pick)

    return client_indices

# -------------------------
# Save client data
# -------------------------
def save_client_data(cid, X, y, out_dir, num_images):
    pt_path = os.path.join(out_dir, f"client_{cid}.pt")
    torch.save((X, y), pt_path)

    img_dir = os.path.join(out_dir, f"client_{cid}_images")
    os.makedirs(img_dir, exist_ok=True)

    for i in range(min(num_images, len(X))):
        img = (X[i] * 255).astype(np.uint8)
        label = y[i]

        # Convert to image
        if img.shape[-1] == 3:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img.squeeze(), mode="L")

        pil_img.save(os.path.join(img_dir, f"sample_{i}_label{label}.png"))

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

    # Partition into aligned clients
    client_indices = partition_clients_aligned(X_train, y_train, num_clients=20)

    print("\n[CLIENT Labeling] Placing labels across clients:\n")
    for cid, idxs in enumerate(client_indices):
        X_client, y_client = X_train[idxs], y_train[idxs]
        save_client_data(cid, X_client, y_client, OUT_DIR, num_images=45) # Nice number % 9

    print("\n[DONE] All client datasets saved in clients_data/")

if __name__ == "__main__":
    main()
