import gradio as gr
import pickle
import pandas as pd
import numpy as np
import os

# =========================
# LOAD SAVED MODEL FILES
# =========================
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print(f"Model loaded! Uses {len(selected_features)} features out of {len(feature_names)}")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_faults(csv_file):
    try:
        # Read uploaded CSV
        df = pd.read_csv(csv_file.name)

        # Check if columns match
        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            return None, f"❌ Missing columns: {missing[:5]}... Please upload MW1 format CSV."

        # Keep only needed columns in correct order
        X = df[feature_names]

        # Scale
        X_scaled = scaler.transform(X)

        # Select features GSO chose
        X_selected = X_scaled[:, selected_features]

        # Predict
        predictions = model.predict(X_selected)
        probabilities = model.predict_proba(X_selected)

        # Build results dataframe
        results = df[feature_names].copy()
        results["Fault_Predicted"] = ["🔴 FAULTY" if p == 1 else "🟢 NOT FAULTY" for p in predictions]
        results["Confidence"] = [f"{max(prob)*100:.1f}%" for prob in probabilities]

        # Summary
        total = len(predictions)
        faulty = sum(predictions)
        summary = f"✅ Analysis Complete!\n📊 Total Modules: {total}\n🔴 Faulty: {faulty}\n🟢 Not Faulty: {total - faulty}\n⚠️ Fault Rate: {faulty/total*100:.1f}%"

        return results, summary

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="GSO Fault Predictor") as demo:
    gr.Markdown("# 🐍 GSO-Based Software Fault Predictor")
    gr.Markdown("Upload a CSV file with software module metrics to predict fault-prone modules.")

    with gr.Row():
        file_input = gr.File(label="📂 Upload CSV File", file_types=[".csv"])
        summary_output = gr.Textbox(label="📋 Summary", lines=6)

    predict_btn = gr.Button("🔍 Predict Faults", variant="primary")
    table_output = gr.Dataframe(label="📊 Prediction Results")

    predict_btn.click(
        fn=predict_faults,
        inputs=file_input,
        outputs=[table_output, summary_output]
    )

    gr.Markdown("**Model:** QDA with GSO Feature Selection | **Dataset:** MW1 | **Accuracy:** 87.11%")

demo.launch()