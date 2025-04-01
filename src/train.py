import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Load data
train = pd.read_csv("data/processed/train.csv")
X_train = train.drop(["customerID", "Churn"], axis=1)
y_train = train["Churn"]

# One-hot encode categoricals
X_train = pd.get_dummies(X_train)

# Start MLflow run
mlflow.set_experiment("Customer_Churn_Prediction")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=params["train"]["random_state"]
    )
    model.fit(X_train, y_train)

    # Log params & metrics
    mlflow.log_params(params["train"])
    train_pred = model.predict(X_train)
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_pred))
    mlflow.log_metric("train_f1", f1_score(y_train, train_pred))

    # Log model
    mlflow.sklearn.log_model(model, "model")
    # At the end of your training script
import json
with open("metrics.json", "w") as f:
    json.dump({
        "train_accuracy": accuracy_score(y_train, train_pred),
        "train_f1": f1_score(y_train, train_pred)
    }, f)


    # Edit src/train.py to ensure it saves to model.pkl
# Add this at the end of your train.py:
import joblib
joblib.dump(model, 'model.pkl')

# And make sure it writes metrics.json
with open('metrics.json', 'w') as f:
    json.dump({
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }, f)