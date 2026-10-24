import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import great_expectations as gx
import pickle

# Force single-threaded CPU execution and stability settings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title="Multimodal Ensemble Skin Cancer Detector", layout="wide")

st.title("🩺 Multimodal Ensemble Skin Cancer Detector (Batch Mode)")
st.write("Upload multiple lesion images (named by their `isic_id`) and a batch CSV file to run automated validation and multi-model ensemble risk evaluation.")

# --- 1. Load Models & Scaler ---
@st.cache_resource
def load_artifacts():
    models_dict = {}
    
    # Check if local absolute path exists 
    if os.path.exists(r"C:\clean_models"):
        base_path = r"C:\clean_models"
    else:
        base_path = "clean_models"

    model_paths = {
        "Baseline MobileNetV2": os.path.join(base_path, "baseline_mobilenetv2"),
        "ResNet50 Multimodal": os.path.join(base_path, "resnet50_multimodal"),
        "Research EfficientNetB0": os.path.join(base_path, "research_efficientnetb0")
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            models_dict[name] = tf.keras.models.load_model(path)
            
    scaler_path = os.path.join(base_path, "scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
    return models_dict, scaler

models, scaler = load_artifacts()

# --- 2. File Uploaders (Multiple Images & Batch CSV) ---
col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_images = st.file_uploader(
        "1. Upload All Lesion Images (e.g., ISIC_0015845.jpg)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
with col_up2:
    uploaded_csv = st.file_uploader("2. Upload Batch CSV", type=["csv"])

# Map uploaded images by filename (stripping extensions) to match isic_id
image_dict = {}
if uploaded_images:
    image_dict = {os.path.splitext(img.name)[0].strip(): img for img in uploaded_images}
    st.info(f"Loaded **{len(image_dict)}** image(s) into memory.")

# --- 3. Expected 46-Feature Schema ---
expected_features = [
    "age_approx", "clin_size_long_diam_mm", "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B", 
    "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext", "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L", 
    "tbp_lv_Lext", "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean", 
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm", 
    "tbp_lv_eccentricity", "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence", "tbp_lv_norm_border", 
    "tbp_lv_norm_color", "tbp_lv_perimeterMM", "tbp_lv_radial_color_std_max", "tbp_lv_stdL", 
    "tbp_lv_stdLExt", "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle", "tbp_lv_x", "tbp_lv_y", "tbp_lv_z",
    "sex_male", "anatom_site_general_head/neck", "anatom_site_general_lower extremity", 
    "anatom_site_general_posterior torso", "anatom_site_general_upper extremity", 
    "tbp_lv_location_simple_Left Arm", "tbp_lv_location_simple_Left Leg", 
    "tbp_lv_location_simple_Right Arm", "tbp_lv_location_simple_Right Leg", 
    "tbp_lv_location_simple_Torso Back", "tbp_lv_location_simple_Torso Front", 
    "tbp_lv_location_simple_Unknown"
]

# --- 4. Execution Trigger ---
if uploaded_images and uploaded_csv is not None:
    df_meta = pd.read_csv(uploaded_csv)
    
    if st.button("Run Batch Ensemble Evaluation"):
        with st.spinner("Validating data with Great Expectations & executing batch models..."):
            
            # --- GREAT EXPECTATIONS VALIDATION (v0.18.12 Corrected) ---
            df_encoded = pd.get_dummies(df_meta)
            df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)
            
            context = gx.get_context(mode="ephemeral")
            datasource = context.sources.add_pandas("pandas_datasource")
            asset = datasource.add_dataframe_asset(name="batch_asset")
            batch_request = asset.build_batch_request(dataframe=df_aligned)
            
            validator = context.get_validator(
                batch_request=batch_request
            )
            validator.expect_table_columns_to_match_ordered_list(expected_features)
            validator.expect_table_row_count_to_be_between(min_value=1, max_value=1000)
            validation_results = validator.validate()
            
            if not validation_results["success"]:
                st.error("❌ Great Expectations Validation Failed! Incoming schema does not match model expectations.")
                st.json(validation_results)
                st.stop()
            else:
                st.success("✅ Great Expectations Validation Passed: Batch schema matches model requirements perfectly!")

            # --- SCALE BATCH DATA ---
            raw_tab_data = df_aligned.values.astype(np.float32)
            if scaler is not None:
                try:
                    tab_array_batch = scaler.transform(raw_tab_data)
                except Exception:
                    tab_array_batch = raw_tab_data
            else:
                tab_array_batch = raw_tab_data

            # --- RUN BATCH PREDICTIONS WITH isic_id IMAGE MAPPING ---
            batch_results = []
            threshold = 0.3  # Union voting threshold
            
            for idx in range(len(tab_array_batch)):
                single_tab_input = np.expand_dims(tab_array_batch[idx], axis=0)
                
                # Safely pull exact isic_id value using column header or fallback to index 0
                if "isic_id" in df_meta.columns:
                    isic_id_val = str(df_meta.iloc[idx]["isic_id"]).strip()
                else:
                    isic_id_val = str(df_meta.iloc[idx, 0]).strip()
                
                # Strictly match image file name to the isic_id value
                if isic_id_val in image_dict:
                    img = Image.open(image_dict[isic_id_val]).resize((109, 109))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                else:
                    batch_results.append({
                        "isic_id": isic_id_val,
                        "Union Verdict": f"❌ Missing Image for {isic_id_val}",
                        **{f"{m}_Prob": "N/A" for m in models.keys()}
                    })
                    continue
                
                row_model_votes = {}
                is_malignant_union = False
                
                # Evaluate across all models in the ensemble
                for name, model in models.items():
                    pred_prob = float(model.predict([img_array, single_tab_input], verbose=0)[0][0])
                    is_flagged = pred_prob >= threshold
                    row_model_votes[name] = pred_prob
                    if is_flagged:
                        is_malignant_union = True
                
                batch_results.append({
                    "isic_id": isic_id_val,
                    "Union Verdict": "Malignant Risk" if is_malignant_union else "Benign / Low Risk",
                    **{f"{m}_Prob": f"{p*100:.2f}%" for m, p in row_model_votes.items()}
                })
            
            # --- DISPLAY RESULTS TABLE ---
            st.divider()
            st.subheader("📊 Batch Evaluation Results Table")
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)