import os
import itertools
import tensorflow as tf
from keras import layers, models
import mlflow
import mlflow.tensorflow
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

# Force safe CPU threading to avoid Windows deadlocks
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

# Set up MLflow local tracking experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ResNet50_Multimodal_GridSearch")

# --- 1. Define Hyperparameter Grid ---
param_grid = {
    "learning_rate": [1e-4, 5e-5],
    "dense_units": [128, 256],
    "dropout_rate": [0.3, 0.5]
}

keys, values = zip(*param_grid.items())
grid_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

print(f"[*] Starting Grid Search: {len(grid_combinations)} hyperparameter combinations found.")

# Load datasets
from model_training_helper_codes import load_multimodal_datasets
train_ds, test_ds = load_multimodal_datasets(test_size=500)

def build_resnet_multimodal(lr, dense_units, dropout_rate, num_tabular_features=46):
    image_input = layers.Input(shape=(109, 109, 3), name="image_input")
    base_cnn = tf.keras.applications.ResNet50(input_shape=(109, 109, 3), include_top=False, weights="imagenet", pooling="avg")
    
    x_img = base_cnn(image_input)
    x_img = layers.Dense(dense_units, activation="swish")(x_img)

    tabular_input = layers.Input(shape=(num_tabular_features,), name="tabular_input")
    x_tab = layers.Dense(dense_units // 2, activation="swish")(tabular_input)

    combined = layers.Concatenate()([x_img, x_tab])
    x_fuse = layers.Dense(dense_units // 2, activation="swish")(combined)
    x_fuse = layers.Dropout(dropout_rate)(x_fuse)
    output = layers.Dense(1, activation="sigmoid")(x_fuse)

    model = models.Model(inputs=[image_input, tabular_input], outputs=output, name="ResNet50_GridSearch_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# Track best model for clean export
best_auc = -1.0
best_model = None
best_params = None

# --- 2. Run Grid Search Loop ---
for i, params in enumerate(grid_combinations, 1):
    print(f"\n--- Training Combination {i}/{len(grid_combinations)} ---")
    print(f"Params: {params}")
    
    with mlflow.start_run(run_name=f"ResNet50_LR_{params['learning_rate']}_Units_{params['dense_units']}"):
        mlflow.log_params(params)
        
        model = build_resnet_multimodal(
            lr=params["learning_rate"],
            dense_units=params["dense_units"],
            dropout_rate=params["dropout_rate"]
        )
        
        # Train model
        model.fit(train_ds, epochs=5, verbose=1)
        
        # Evaluate on Test Set
        y_true = []
        y_pred_probs = []
        for batch in test_ds:
            inputs, labels = batch
            preds = model.predict(inputs, verbose=0).flatten()
            y_pred_probs.extend(preds)
            if hasattr(labels, "numpy"):
                y_true.extend(labels.numpy())
            else:
                y_true.extend(labels)
                
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred_binary = (y_pred_probs >= 0.4).astype(int)
        
        test_auc = roc_auc_score(y_true, y_pred_probs)
        test_precision = precision_score(y_true, y_pred_binary, zero_division=0)
        test_recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        
        print(f"[+] Results -> AUC: {test_auc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")
        
        # Log model artifact to MLflow
        mlflow.tensorflow.log_model(model, "model")
        
        # Track if this is the best model so far
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
            best_params = params

# --- 3. Automatically Save the Best Model to Clean Folder ---
if best_model is not None:
    dest_path = r"C:\clean_models\resnet50_multimodal"
    print(f"\n[✨] Best model found! AUC: {best_auc:.4f} with params: {best_params}")
    print(f"[*] Saving best model directly to clean notebook path: {dest_path}")
    
    # Overwrite the old ResNet50 model folder in C:\clean_models with the newly tuned version
    if os.path.exists(dest_path):
        import shutil
        shutil.rmtree(dest_path)
        
    best_model.save(dest_path)
    print("[+] Successfully updated C:\\clean_models\\resnet50_multimodal! Your notebook will now load this tuned version automatically.")

print("\n[+] Grid Search completed successfully!")