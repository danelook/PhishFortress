import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Script has started running...")

# Load and preprocess the dataset
csv_path = '../data/emails.csv'

# Check if the file exists
if not os.path.exists(csv_path):
    print(f"Error: The file {csv_path} does not exist.")
    exit()

# Load the dataset
df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")  # Print dataset shape

print("Preprocessing data...")

# Remove NaN values from 'Email Text' and 'Email Type'
df.dropna(subset=['Email Text', 'Email Type'], inplace=True)

# Remove rows with empty strings in 'Email Text'
df = df[df['Email Text'].str.strip() != '']

# Preprocess the data (email body and labels)
X = df['Email Text']  # Email text
y = df['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)  # Labels (1 for phishing, 0 for legitimate)

# Convert the email text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

print("Data preprocessing completed!")

print("Splitting the data...")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model training completed!")

print("Evaluating the model...")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn for better visuals
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model and vectorizer
print("Saving model and vectorizer...")
joblib.dump(model, '../models/phishing_model.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')
print("Model and vectorizer saved successfully!")