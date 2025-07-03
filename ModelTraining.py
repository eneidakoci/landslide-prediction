import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load dataset
df = pd.read_csv(r"C:\Users\User\Desktop\cleaned_training_dataset_rounded.csv")

# 2. Drop duplicates
df = df.drop_duplicates()
print(f" Removed duplicates. Dataset now has {len(df)} rows.")

# 3. Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Check class balance
print("Class distribution:\n", df['Class'].value_counts())
sns.countplot(data=df, x='Class')
plt.title("Class Distribution")
plt.show()

# 5. Check duplicates again (for potential leakage)
dupes = df.duplicated().sum()
print(f" Duplicated rows in dataset: {dupes}")

# 6. Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

overlapping = pd.merge(X_train, X_test, how='inner')
print(f" Overlapping samples between train/test: {len(overlapping)}")

# 8. Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Evaluation function
model_scores = {}

def evaluate_model(name, model, X_test, y_test, scaled=False):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred) # Correct predictions overall
    prec = precision_score(y_test, y_pred) #Of the predicted landslides, how  many were correct
    rec = recall_score(y_test, y_pred) #how many actual landslides were caught
    f1 = f1_score(y_test, y_pred) # Mean of precision and recall

    model_scores[name] = {
        "model": model,
        "scaled": scaled,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    print(f"\n {name} Evaluation:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{name} - ROC Curve")
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{name} - Precision-Recall Curve")
        plt.show()

# 10. Cross-validation
def cross_validate_model(name, model, X, y, cv=5, scale=False):
    if scale:
        from sklearn.pipeline import make_pipeline
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f" {name} Cross-Validation Accuracy ({cv}-fold): {scores.mean():.4f} ± {scores.std():.4f}")

# 11. Train and evaluate models

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf, X_test, y_test)
cross_validate_model("Random Forest", rf, X, y)

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Random Forest - Top 10 Feature Importances")
plt.show()

# SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
evaluate_model("SVM", svm, X_test_scaled, y_test, scaled=True)
cross_validate_model("SVM", svm, X, y, scale=True)

# XGBoost
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
evaluate_model("XGBoost", xgb, X_test, y_test)
cross_validate_model("XGBoost", xgb, X, y)


# 12. Identify Best Model
best_by_f1 = max(model_scores.items(), key=lambda x: x[1]['f1_score'])
print("\n Best Model by F1 Score:", best_by_f1[0])
print("F1 Score:", best_by_f1[1]['f1_score'])

# 13. Comparison Chart
score_df = pd.DataFrame.from_dict(model_scores, orient='index')
score_df = score_df[["accuracy", "precision", "recall", "f1_score"]]
score_df.plot(kind='bar', figsize=(12, 6), colormap='Set2')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 14. Save models
joblib.dump(rf, r"random_forest_model.pkl")
joblib.dump(svm, r"svm_model.pkl")
joblib.dump(xgb, r"xgboost_model.pkl")
joblib.dump(scaler, r"scaler.pkl")
print("\n All models and scaler saved.")
