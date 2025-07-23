# Step 1: Upload the dataset file interactively
from google.colab import files
uploaded = files.upload()

import io
import pandas as pd

# Step 2: Load the uploaded dataset into a DataFrame
data = pd.read_csv(io.BytesIO(uploaded['creditcard.csv']))
print(f"Dataset shape: {data.shape}")
print(data['Class'].value_counts())

# Step 3: Preprocessing and modeling code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocess data function
def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']

    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"Before SMOTE:\n{y_train.value_counts()}")
    print(f"After SMOTE:\n{y_train_res.value_counts()}")

    return X_train_res, X_test, y_train_res, y_test

# Train models function
def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf

# Evaluate models function
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Confusion Matrix:\n", cm)

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"{model_name} ROC-AUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Run the full pipeline
X_train, X_test, y_train, y_test = preprocess_data(data)
lr_model, rf_model = train_models(X_train, y_train)

evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
evaluate_model(rf_model, X_test, y_test, "Random Forest")

