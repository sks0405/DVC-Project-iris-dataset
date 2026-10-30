import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
 
# ── Step 1: Load dataset ──────────────────────────────────
data_path = os.path.join("data", "iris.csv")
df = pd.read_csv(data_path)
 
# ── Step 2: Split into features (X) and target (y) ───────
X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df['class']
 
# ── Step 3: Train / Test split (80% train, 20% test) ─────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# ── Step 4: Train Decision Tree model ────────────────────
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
 
# ── Step 5: Evaluate the model ───────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy*100:.2f}%")

# ── Step 6: Save the trained model ───────────────────────
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "iris_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully to: {model_path}")
