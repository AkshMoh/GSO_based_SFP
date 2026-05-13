import gradio as gr
import numpy as np
import pandas as pd
import random
import math
import json
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =========================
# GSO FUNCTIONS
# =========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def binarise(position):
    prob = sigmoid(position)
    return (np.random.rand(*position.shape) < prob).astype(int)

def levy_flight(dimensions, beta=1.5):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (num / den) ** (1.0 / beta)
    u = np.random.randn(dimensions) * sigma
    v = np.random.randn(dimensions)
    return u / (np.abs(v) ** (1.0 / beta))

def fitness(feat, label, selected_indices, model_type):
    if len(selected_indices) == 0:
        return 1.0
    X_tr, X_val, y_tr, y_val = train_test_split(
        feat[:, selected_indices], label,
        test_size=0.2, random_state=42, stratify=label
    )
    if model_type == "KNN":
        clf = KNeighborsClassifier()
    elif model_type == "NB":
        clf = GaussianNB()
    elif model_type == "QDA":
        clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    else:
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    try:
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        return float(np.sum(preds != y_val)) / len(y_val)
    except:
        return 1.0

def GSO(feat, label, sol_count, dimensions, iterations_count,
        lower_bound, upper_bound, model_type):
    positions = lower_bound + np.random.rand(sol_count, dimensions) * (upper_bound - lower_bound)
    fit_vals = np.zeros(sol_count)
    bin_pop = np.zeros_like(positions, dtype=int)

    for s in range(sol_count):
        bin_pop[s] = binarise(positions[s])
        fit_vals[s] = fitness(feat, label, np.where(bin_pop[s] == 1)[0], model_type)

    idx = np.argsort(fit_vals)
    positions = positions[idx]
    bin_pop = bin_pop[idx]
    fit_vals = fit_vals[idx]

    fitG = fit_vals[0]
    Xgb_bin = bin_pop[0].copy()
    Xgb_cont = positions[0].copy()

    for t in range(iterations_count):
        A = 1.0 - t / iterations_count
        for s in range(sol_count):
            if random.random() < 0.3:
                L = levy_flight(dimensions)
                positions[s] = Xgb_cont + L * A * (Xgb_cont - positions[s])
            else:
                pred_idx = (s - 1) % sol_count
                positions[s] += A * random.random() * (Xgb_cont - positions[s]) + \
                                A * random.random() * (positions[pred_idx] - positions[s])
            positions[s] = np.clip(positions[s], lower_bound, upper_bound)
            bin_pop[s] = binarise(positions[s])
            fit_vals[s] = fitness(feat, label, np.where(bin_pop[s] == 1)[0], model_type)
            if fit_vals[s] < fitG:
                fitG = fit_vals[s]
                Xgb_bin = bin_pop[s].copy()
                Xgb_cont = positions[s].copy()

    return Xgb_bin

# =========================
# MAIN PREDICTION FUNCTION
# =========================
def predict_faults(csv_file, model_choice, iterations):
    try:
        # Read CSV
        df = pd.read_csv(csv_file.name)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if df.shape[1] < 2:
            return None, "❌ CSV must have at least 2 columns (features + target)"

        # Last column = target
        feature_cols = list(df.columns[:-1])
        target_col = df.columns[-1]

        X_raw = df[feature_cols].values.astype(float)
        y_raw = df[target_col].astype(str).str.strip().str.lower()
        y = np.where(y_raw.isin(['true', 'yes', '1', 'y']), 1, 0)

        if len(np.unique(y)) < 2:
            return None, "❌ Target column must have both faulty (1) and not-faulty (0) samples"

        # Scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )

        # Run GSO
        best_sol = GSO(
            X_train, y_train,
            sol_count=10,
            dimensions=X_scaled.shape[1],
            iterations_count=int(iterations),
            lower_bound=-4,
            upper_bound=4,
            model_type=model_choice
        )

        selected_idx = np.where(best_sol == 1)[0]
        if len(selected_idx) == 0:
            selected_idx = np.array([0])

        # Train final model
        if model_choice == "KNN":
            clf = KNeighborsClassifier()
        elif model_choice == "NB":
            clf = GaussianNB()
        elif model_choice == "QDA":
            clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
        else:
            clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)

        clf.fit(X_train[:, selected_idx], y_train)
        pred = clf.predict(X_test[:, selected_idx])

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)

        # Predict all rows
        all_pred = clf.predict(X_scaled[:, selected_idx])
        fault_count = int(sum(all_pred))

        # Results dataframe
        results = df[feature_cols].copy()
        results["Fault_Predicted"] = ["🔴 FAULTY" if p == 1 else "🟢 NOT FAULTY" for p in all_pred]

        selected_names = [feature_cols[i] for i in selected_idx]
        reduction = (1 - len(selected_idx)/len(feature_cols)) * 100

        summary = f"""✅ Analysis Complete!

📊 Dataset Info:
   Total Modules  : {len(df)}
   Total Features : {len(feature_cols)}

🧬 GSO Feature Selection:
   Selected Features : {len(selected_idx)} / {len(feature_cols)}
   Reduction         : {reduction:.1f}%
   Features Used     : {', '.join(selected_names[:5])}{'...' if len(selected_names) > 5 else ''}

🎯 Model Performance ({model_choice}):
   Accuracy  : {acc*100:.2f}%
   Precision : {prec:.4f}
   Recall    : {rec:.4f}
   F1-Score  : {f1:.4f}

🔴 Fault Prediction:
   Faulty     : {fault_count}
   Not Faulty : {len(df) - fault_count}
   Fault Rate : {fault_count/len(df)*100:.1f}%"""

        return results, summary

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="GSO Fault Predictor") as demo:
    gr.Markdown("# 🐍 GSO-Based Software Fault Predictor")
    gr.Markdown("Upload **any** software metrics CSV. GSO will automatically select optimal features and predict fault-prone modules.")
    gr.Markdown("**Format:** Any columns as features, last column as target (1/True/Y = faulty, 0/False/N = not faulty)")

    with gr.Row():
        file_input = gr.File(label="📂 Upload CSV", file_types=[".csv"])
        with gr.Column():
            model_choice = gr.Dropdown(
                choices=["QDA", "KNN", "NB", "RF"],
                value="QDA",
                label="🤖 Classifier"
            )
            iterations = gr.Slider(
                minimum=5,
                maximum=30,
                value=10,
                step=5,
                label="⚙️ GSO Iterations (more = slower but better)"
            )

    predict_btn = gr.Button("🔍 Run GSO + Predict", variant="primary")

    with gr.Row():
        summary_output = gr.Textbox(label="📋 Summary & Metrics", lines=20)

    table_output = gr.Dataframe(label="📊 Module-wise Predictions")

    predict_btn.click(
        fn=predict_faults,
        inputs=[file_input, model_choice, iterations],
        outputs=[table_output, summary_output]
    )

    gr.Markdown("---")
    gr.Markdown("**Algorithm:** Glider Snake Optimization (GSO) | **Supported:** Any software metrics CSV | **Models:** QDA, KNN, NB, RF")

demo.launch()