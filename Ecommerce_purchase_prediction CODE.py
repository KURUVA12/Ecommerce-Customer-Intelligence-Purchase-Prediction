
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_excel("C:/Users/harik/Downloads/ecommerce_purchase_prediction_dataset (1).xlsx")

print(df.info())
print(df.describe())
print(df.head())
print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace(
        {"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA}
    )

df["purchase"] = pd.to_numeric(df["purchase"], errors="coerce")
df = df.dropna(subset=["purchase"])
df = df[df["purchase"].isin([0, 1])]
df["purchase"] = df["purchase"].astype(int)

for col in df.select_dtypes(include=np.number).columns:
    if col != "purchase":
        df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object", "string"]).columns:
    mode = df[col].mode(dropna=True)
    if len(mode) > 0:
        df[col] = df[col].fillna(mode.iloc[0])

if df["session_date"].isna().any() and df["session_date"].notna().any():
    df["session_date"] = df["session_date"].fillna(
        df["session_date"].dropna().median()
    )

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("ecommerce_purchase_prediction_cleaned.csv", index=False)


plt.figure(figsize=(7, 4))
sns.countplot(
    data=df, x="purchase", hue="purchase",
    palette="Set2", legend=False
)
plt.title("Purchase Distribution")
plt.xlabel("Purchase (0 = No, 1 = Yes)")
plt.ylabel("Number of Sessions")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


num_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(11, 8))
sns.heatmap(
    num_df.corr(),
    annot=True,
    cmap="turbo",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 5))
sns.barplot(
    data=df,
    x="product_category",
    y="purchase",
    hue="product_category",
    palette="bright",
    legend=False,
    errorbar=None
)
plt.title("Purchase Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="purchase",
    y="product_price",
    hue="purchase",
    palette="Set3",
    legend=False
)
plt.title("Product Price by Purchase Outcome")
plt.xlabel("Purchase (0 = No, 1 = Yes)")
plt.ylabel("Product Price")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="page_views",
    y="session_duration_seconds",
    hue="purchase",
    palette="bright",
    alpha=0.7
)
plt.title("Page Views vs Session Duration")
plt.xlabel("Page Views")
plt.ylabel("Session Duration (Seconds)")
plt.tight_layout()
plt.show()


purchase_rate = df.groupby("device_type")["purchase"].mean()

plt.figure(figsize=(7, 4))
purchase_rate.plot(
    kind="bar",
    color=sns.color_palette("husl", len(purchase_rate))
)
plt.title("Purchase Rate by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


purchase_rate = df.groupby("customer_type")["purchase"].mean()

plt.figure(figsize=(7, 4))
purchase_rate.plot(
    kind="bar",
    color=sns.color_palette("Set1", len(purchase_rate))
)
plt.title("Purchase Rate by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


X = df.drop(
    columns=[
        col for col in [
            "session_id",
            "user_id",
            "product_id",
            "session_date",
            "purchase",
            "added_to_cart"
        ]
        if col in df.columns
    ]
)

X = pd.get_dummies(X, drop_first=True)
y = df["purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ])
}


results = []
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(
            y_test,
            model.predict_proba(X_test)[:, 1]
        )
    except Exception:
        auc = np.nan

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))


results = pd.DataFrame(results)
results = results.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nModel Comparison:")
print(results)

results.to_csv("model_results.csv", index=False)

best_model_name = results.iloc[0]["Model"]
best_prediction = predictions[best_model_name]

print("\nBest Model by F1 Score:", best_model_name)


plt.figure(figsize=(9, 5))
sns.barplot(
    data=results,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette="bright",
    legend=False
)
plt.title("Machine Learning Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 5))
sns.barplot(
    data=results,
    x="Model",
    y="F1 Score",
    hue="Model",
    palette="Set1",
    legend=False
)
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, best_prediction),
    annot=True,
    fmt="d",
    cmap="turbo"
)
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.tight_layout()
plt.show()


rng = np.random.default_rng(42)

plt.figure(figsize=(7, 5))
plt.scatter(
    y_test.to_numpy() + rng.normal(0, 0.025, len(y_test)),
    best_prediction + rng.normal(0, 0.025, len(best_prediction)),
    color="royalblue",
    alpha=0.6,
    s=30
)
plt.plot([-0.1, 1.1], [-0.1, 1.1], "r--")
plt.xticks([0, 1], ["No Purchase", "Purchase"])
plt.yticks([0, 1], ["Predicted No", "Predicted Purchase"])
plt.xlabel("Actual Purchase")
plt.ylabel("Predicted Purchase")
plt.title("Actual vs Predicted Purchase - " + best_model_name)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print("\nPurchase Rate:")
print(df["purchase"].value_counts(normalize=True))

print("\nProject Analysis Completed Successfully!")
