import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_multimodal_datasets(
    csv_path=r"C:\Users\Aim\OneDrive\Рабочий стол\skin cancer detection\data\metadata_preprocessed.csv",
    ground_truth_path=r"C:\Users\Aim\OneDrive\Рабочий стол\skin cancer detection\data\ISIC_2024_Training_GroundTruth.csv",
    test_size=2000,
    malignant_ratio=0.05,
    batch_size=32,
    max_train_samples=8000,
):
  """Loads metadata and ground truth labels, merges them,

  verifies disk assets, creates a stratified test set, keeps ALL
  remaining malignant cases in training, and caps benign samples.
  """
  print("[+] Loading metadata and ground truth labels...")
  df_meta = pd.read_csv(csv_path)
  df_gt = pd.read_csv(ground_truth_path)

  # Merge datasets on unique patient/image identifier
  df = pd.merge(df_meta, df_gt, on="isic_id", how="inner")

  # Define base image directory using absolute path
  img_dir = r"C:\Users\Aim\OneDrive\Рабочий стол\skin cancer detection\data\images"

  # Verify image files physically exist to prevent runtime crashes
  print("[+] Verifying image assets on disk...")
  valid_mask = df["isic_id"].apply(
      lambda img_id: os.path.exists(os.path.join(img_dir, f"{img_id}.jpg"))
  )
  df = df[valid_mask].reset_index(drop=True)
  print(f"[+] Valid samples with verified images: {len(df)}")

  # Stratified test set isolation (2000 samples with exact malignant ratio)
  n_test_malignant = int(test_size * malignant_ratio)
  n_test_benign = test_size - n_test_malignant

  df_malignant = df[df["malignant"] == 1]
  df_benign = df[df["malignant"] == 0]

  test_malignant = df_malignant.sample(
      n=min(n_test_malignant, len(df_malignant)), random_state=42
  )
  test_benign = df_benign.sample(
      n=min(n_test_benign, len(df_benign)), random_state=42
  )

  df_test = pd.concat([test_malignant, test_benign]).sample(
      frac=1, random_state=42
  )

  # Remainder used for training pool
  df_train_pool = df.drop(df_test.index)

  # Separate training pool into malignant and benign classes
  train_malignant = df_train_pool[df_train_pool["malignant"] == 1]
  train_benign = df_train_pool[df_train_pool["malignant"] == 0]

  # KEEP ALL REMAINING MALIGNANT CASES, downsample benign only if needed
  if max_train_samples and len(df_train_pool) > max_train_samples:
    n_benign_needed = max_train_samples - len(train_malignant)
    if n_benign_needed > 0 and len(train_benign) > n_benign_needed:
      train_benign = train_benign.sample(n=n_benign_needed, random_state=42)

  # Combine and shuffle training set
  df_train = (
      pd.concat([train_malignant, train_benign])
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
  )
  df_test = df_test.reset_index(drop=True)

  print(
      f"[+] Train set: {len(df_train)} samples | Malignant ratio:"
      f" {df_train['malignant'].mean()*100:.2f}%"
  )
  print(
      f"[+] Test set: {len(df_test)} samples | Malignant ratio:"
      f" {df_test['malignant'].mean()*100:.2f}%"
  )

  def process_split(df_subset, is_training=True):
    X_tab = (
        df_subset.drop(
            columns=["isic_id", "patient_id", "malignant"], errors="ignore"
        )
        .values.astype(np.float32)
    )
    y_labels = df_subset["malignant"].values.astype(np.float32)

    image_paths = [
        os.path.join(img_dir, f"{img_id}.jpg")
        for img_id in df_subset["isic_id"].values
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        ((image_paths, X_tab), y_labels)
    )

    def load_and_preprocess_image(inputs, label):
      path, tab_features = inputs
      image_string = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image_string, channels=3)
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, [109, 109])

      # --- Safe, Standardized Data Augmentation for Training ---
      if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)

      return {"image_input": image, "tabular_input": tab_features}, label

    dataset = dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
      dataset = (
          dataset.shuffle(buffer_size=min(len(df_subset), 1000), seed=42)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
      )
    else:
      dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

  train_dataset = process_split(df_train, is_training=True)
  test_dataset = process_split(df_test, is_training=False)

  return train_dataset, test_dataset


class WarmupCosineDecay(tf.keras.callbacks.Callback):
  """Research-grade learning rate scheduler with linear warmup followed by cosine decay."""

  def __init__(
      self,
      warmup_epochs=1,
      total_epochs=20,
      peak_lr=2e-4,
      min_lr=1e-6,
  ):
    super(WarmupCosineDecay, self).__init__()
    self.warmup_epochs = warmup_epochs
    self.total_epochs = total_epochs
    self.peak_lr = peak_lr
    self.min_lr = min_lr

  def on_epoch_begin(self, epoch, logs=None):
    if epoch < self.warmup_epochs:
      lr = self.peak_lr * (epoch + 1) / self.warmup_epochs
    else:
      progress = (epoch - self.warmup_epochs) / (
          self.total_epochs - self.warmup_epochs
      )
      lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
          1 + np.cos(np.pi * progress)
      )

    if self.model and self.model.optimizer:
      tf.keras.backend.set_value(self.model.optimizer.lr, lr)
    print(f"\n[LR Scheduler] Epoch {epoch + 1}: Learning rate -> {lr:.6f}")