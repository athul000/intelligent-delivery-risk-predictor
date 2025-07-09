import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("D:\Project\intelligent-delivery-risk-predictor\data\Delivery risk predictor (Athul project).csv")
print("✅ Loaded data with shape:", df.shape)

# Check class distribution
print("\n Target column distribution:")
print(df['predicted_risk'].value_counts())

# First split — 60% for model training, 40% for final unseen test
X = df.drop("predicted_risk", axis=1)
y = df["predicted_risk"]

X_trainval, X_unseen, y_trainval, y_unseen = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

print("\n 60% training+validation shape:", X_trainval.shape)
print(" 40% unseen test shape:", X_unseen.shape)

# Split 60% further — 80% train, 20% val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
)

print("\n Final Split Sizes:")
print(" Training:", X_train.shape)
print(" Validation:", X_val.shape)
print(" Unseen Test:", X_unseen.shape)

# Save as CSV files
df_train = X_train.copy()
df_train["predicted_risk"] = y_train
df_train.to_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\train_projects.csv", index=False)

df_val = X_val.copy()
df_val["predicted_risk"] = y_val
df_val.to_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\val_projects.csv", index=False)

df_unseen = X_unseen.copy()
df_unseen["predicted_risk"] = y_unseen
df_unseen.to_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\unseen_projects.csv", index=False)

print("\n✅ Saved all files to /data:")
print(" - train_projects.csv")
print(" - val_projects.csv")
print(" - unseen_projects.csv")
