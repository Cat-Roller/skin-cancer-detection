import os

# Suppress oneDNN floating-point round-off warning clutter
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from keras import layers, models
from model_training_helper_codes import (
    WarmupCosineDecay,
    load_multimodal_datasets,
)

# --- 1. MLflow Setup with Pure Autologging (Disabling schema warning) ---
mlflow.set_experiment("skin_cancer_multimodal_resnet50")
mlflow.tensorflow.autolog(log_datasets=False)


# --- 2. Multimodal Functional API Model Definition (ResNet50 Backbone) ---
def create_resnet_multimodal_model(num_tabular_features):
  image_input = layers.Input(shape=(109, 109, 3), name="image_input")

  # Using ResNet50 backbone (pretrained ImageNet weights for robust feature extraction)
  base_cnn = tf.keras.applications.ResNet50(
      input_shape=(109, 109, 3),
      include_top=False,
      weights="imagenet",
      pooling="avg",
  )
  x_img = base_cnn(image_input)
  x_img = layers.Dense(128, activation="relu", name="img_embedding")(x_img)

  tabular_input = layers.Input(shape=(num_tabular_features,), name="tabular_input")
  x_tab = layers.Dense(64, activation="relu")(tabular_input)
  x_tab = layers.Dropout(0.3)(x_tab)
  x_tab = layers.Dense(32, activation="relu", name="tab_embedding")(x_tab)

  combined = layers.Concatenate(name="fusion_layer")([x_img, x_tab])
  x_fuse = layers.Dense(64, activation="relu")(combined)
  x_fuse = layers.Dropout(0.3)(x_fuse)
  output = layers.Dense(1, activation="sigmoid", name="malignant_output")(x_fuse)

  model = models.Model(
      inputs=[image_input, tabular_input],
      outputs=output,
      name="ResNet50_Multimodal_Classifier",
  )
  return model


# --- 3. Training Pipeline Execution ---
def run_resnet_training():
  BATCH_SIZE = 32
  EPOCHS = 20
  WARMUP_EPOCHS = 2
  PEAK_LR = 1e-4
  NUM_TABULAR_FEATURES = 46
  CSV_PATH = r"C:\Users\Aim\OneDrive\Рабочий стол\skin cancer detection\data\metadata_preprocessed.csv"
  GT_PATH = r"C:\Users\Aim\OneDrive\Рабочий стол\skin cancer detection\data\ISIC_2024_Training_GroundTruth.csv"

  with mlflow.start_run(run_name="ResNet50_Multimodal_Baseline"):
    # Load datasets using helper (8k train, all malignant kept, 2k test)
    train_dataset, test_dataset = load_multimodal_datasets(
        csv_path=CSV_PATH,
        ground_truth_path=GT_PATH,
        test_size=2000,
        malignant_ratio=0.05,
        batch_size=BATCH_SIZE,
        max_train_samples=8000,
    )

    model = create_resnet_multimodal_model(NUM_TABULAR_FEATURES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PEAK_LR),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    # Callbacks
    lr_scheduler = WarmupCosineDecay(
        warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS, peak_lr=PEAK_LR
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=4, restore_best_weights=True
    )

    print(
        "\n[+] ResNet50 Multimodal Pipeline configured. Starting training..."
    )

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=[lr_scheduler, early_stopping],
    )

    print(
        "[+] ResNet50 training complete! Metrics and model logged to MLflow."
    )


if __name__ == "__main__":
  run_resnet_training()