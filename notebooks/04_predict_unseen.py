import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# ML Models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# Load unseen dataset
df_unseen = pd.read_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\unseen_projects.csv")
# Load label encoder
label_encoder = LabelEncoder()
df_unseen["encoded_risk"] = label_encoder.fit_transform(df_unseen["predicted_risk"])
# Preprocess features
cols_to_drop = ['predicted_risk', 'project_id', 'project_name', 'client_name', 'start_date', 'end_date']
X_unseen = df_unseen.drop(columns=cols_to_drop + ['encoded_risk'], errors='ignore').select_dtypes(include='number')
y_true = df_unseen["encoded_risk"]
# Load or re-train model (Logistic Regression)
# You can replace this with your best model like XGBoost if needed
model = LogisticRegression(max_iter=1000)
df_train = pd.read_csv("data/train_projects.csv")
df_train["encoded_risk"] = label_encoder.fit_transform(df_train["predicted_risk"])
X_train = df_train.drop(columns=cols_to_drop + ['encoded_risk'], errors='ignore').select_dtypes(include='number')
y_train = df_train["encoded_risk"]
model.fit(X_train, y_train)
# Predict on unseen data
y_pred = model.predict(X_unseen)
decoded_pred = label_encoder.inverse_transform(y_pred)
decoded_true = label_encoder.inverse_transform(y_true)

# Save predictions
df_unseen["predicted_by_model"] = decoded_pred
df_unseen.to_csv("D:\Project\intelligent-delivery-risk-predictor\data\predicted_unseen_projects.csv", index=False)
print("✅ Predictions saved to: data/predicted_unseen_projects.csv")

# Show performance
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("📄 Classification Report on Unseen Data:\n")
print(classification_report(decoded_true, decoded_pred))

cm = confusion_matrix(decoded_true, decoded_pred, labels=["High", "Medium", "Low"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["High", "Medium", "Low"],
            yticklabels=["High", "Medium", "Low"])
plt.title("Confusion Matrix - Unseen Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()