import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# 1. Load the clean dataset
df = pd.read_csv(r"C:\Users\User\Desktop\cleaned_training_dataset_rounded.csv")

# 2. Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# 3. Add Gaussian noise to all features
np.random.seed(42)
noise_level = 0.05  # 5% noise
X_noisy = X + noise_level * np.random.normal(loc=0, scale=1, size=X.shape)

# 4. Train/test split on noisy data
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Train XGBoost
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# 6. Evaluate
print("\n📊 Evaluation on Noisy Data:")
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
