import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load test data
test = pd.read_csv("data/processed/test.csv")
X_test = test.drop(["customerID", "Churn"], axis=1)
y_test = test["Churn"]
X_test = pd.get_dummies(X_test)

# Load best model from MLflow
best_run = mlflow.search_runs(order_by=["metrics.test_f1 DESC"]).iloc[0]
model = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/model")

# Evaluate
test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test F1 Score:", f1_score(y_test, test_pred))
print(classification_report(y_test, test_pred))