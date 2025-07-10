import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load datasets
df_train = pd.read_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\train_projects.csv")
df_val = pd.read_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\val_projects.csv")

label_encoder = LabelEncoder()
df_train["encoded_risk"] = label_encoder.fit_transform(df_train["predicted_risk"])
df_val["encoded_risk"] = label_encoder.transform(df_val["predicted_risk"])

cols_to_drop = ['predicted_risk', 'project_id', 'project_name', 'client_name', 'start_date', 'end_date']
X_train = df_train.drop(columns=cols_to_drop + ['encoded_risk'], errors='ignore').select_dtypes(include='number')
X_val = df_val.drop(columns=cols_to_drop + ['encoded_risk'], errors='ignore').select_dtypes(include='number')

y_train = df_train["encoded_risk"]
y_val = df_val["encoded_risk"]


# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔵 Training: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    # Decode labels
    decoded_preds = label_encoder.inverse_transform(preds)
    decoded_y_val = label_encoder.inverse_transform(y_val)

    acc = accuracy_score(decoded_y_val, decoded_preds)
    print(f" Accuracy: {acc:.4f}")
    print("📄 Classification Report:\n", classification_report(decoded_y_val, decoded_preds))

    results.append((name, acc))

    # Plot Confusion Matrix
    cm = confusion_matrix(decoded_y_val, decoded_preds, labels=["High", "Medium", "Low"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["High", "Medium", "Low"],
                yticklabels=["High", "Medium", "Low"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ✅ 6. Accuracy Summary
results.sort(key=lambda x: x[1], reverse=True)
print("\n Model Accuracy Summary:")
for name, acc in results:
    print(f"{name}: {acc:.4f}")