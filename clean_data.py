import numpy as np

def clean_data(npz_path="pathmnist.npz", out_path="pathmnist_cleaned.npz"):
    # Load original PathMNIST
    data = np.load(npz_path)
    X_train, y_train = data["train_images"], data["train_labels"].squeeze()
    X_val, y_val = data["val_images"], data["val_labels"].squeeze()
    X_test, y_test = data["test_images"], data["test_labels"].squeeze()

    print(f"[INFO] Original train size: {len(X_train)}")

    # --- Cleaning steps ---
    # 1. Remove blank images
    mask = [img.sum() > 0 for img in X_train]
    X_train, y_train = X_train[mask], y_train[mask]
    print(f"[CLEAN] Removed blanks -> {len(X_train)} samples remain")

    # 2. Remove duplicates
    X_reshaped = X_train.reshape(len(X_train), -1)
    _, unique_idx = np.unique(X_reshaped, axis=0, return_index=True)
    X_train, y_train = X_train[unique_idx], y_train[unique_idx]
    print(f"[CLEAN] Removed duplicates -> {len(X_train)} samples remain")

    # 3. Normalize to [0,1]
    X_train = X_train.astype(np.float32) / 255.0
    X_val   = X_val.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0
    print(f"[CLEAN] Normalized pixel values to [0,1]")

    # Save cleaned dataset
    np.savez(out_path,
             train_images=X_train, train_labels=y_train,
             val_images=X_val, val_labels=y_val,
             test_images=X_test, test_labels=y_test)
    print(f"[DONE] Cleaned dataset saved to {out_path}")

if __name__ == "__main__":
    clean_data()
