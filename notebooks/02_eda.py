import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load train data
df = pd.read_csv(r"D:\Project\intelligent-delivery-risk-predictor\data\train_projects.csv")
print("Shape:", df.shape)
df.head()

# sns.countplot(data=df, x='predicted_risk', palette="Set2")
# plt.title("Distribution of Project Risk Levels")
# plt.show()

df.describe()

# plt.figure(figsize=(12,8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

# numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# for col in numerical_columns:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(data=df, x='predicted_risk', y=col, palette="Set2")
#     plt.title(f"{col} vs Risk Level")
#     plt.show()


df.select_dtypes(include='number').groupby(df["predicted_risk"]).mean().to_csv(
    r"D:\Project\intelligent-delivery-risk-predictor\data\risk_grouped_stats.csv"
)




