import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the vectorizer and datasets
model_dir = r'C:\Users\novem\phishfortress\models'
vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
X_train, y_train = joblib.load(os.path.join(model_dir, 'train_data.pkl'))
X_test, y_test = joblib.load(os.path.join(model_dir, 'test_data.pkl'))

# Convert string labels to integers if necessary
if y_train.dtype != 'int' and y_train.dtype != 'int64':
    unique_labels = y_train.unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Label mapping: {label_mapping}")  # Display the mapping for verification
    y_train = y_train.map(label_mapping)
    y_test = y_test.map(label_mapping)

# Check for any unmapped or missing values after conversion
if y_train.isna().any() or y_test.isna().any():
    raise ValueError("Some labels could not be mapped to integers. Please check your data.")

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, os.path.join(model_dir, 'logistic_regression_model.pkl'))
print("Model saved successfully.")

