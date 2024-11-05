import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load all three datasets
df1 = pd.read_csv(r'C:\Users\novem\phishfortress\data\emails1.csv')
df2 = pd.read_csv(r'C:\Users\novem\phishfortress\data\emails2.csv')
df3 = pd.read_csv(r'C:\Users\novem\phishfortress\data\emails3.csv')

# Combine the datasets into one DataFrame
df = pd.concat([df1, df2, df3], ignore_index=True)

# Check that required columns exist and drop rows with missing values in 'subject' or 'body'
if 'subject' in df.columns and 'body' in df.columns and 'label' in df.columns:
    df.dropna(subset=['subject', 'body', 'label'], inplace=True)
else:
    print("Required columns 'subject', 'body', and/or 'label' not found!")

# Combine the 'subject' and 'body' fields into a single text feature for analysis
df['email_text'] = df['subject'] + ' ' + df['body']

# Optional: drop the columns that are not necessary for the model (you can keep any of these if needed)
df = df.drop(columns=['subject', 'body', 'sender', 'receiver', 'date', 'urls'])

# Splitting the dataset into training and testing sets
X = df['email_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the email text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training a basic Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Create a folder for models if it doesn't exist
model_dir = r'C:\Users\novem\phishfortress\models'
os.makedirs(model_dir, exist_ok=True)

# Overwrite the models and save them under the new names
joblib.dump(model, os.path.join(model_dir, 'phishing_model.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))

# Create the heatmap with Seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])

# Add labels and titles
plt.title('Confusion Matrix of Combined Phish Detection Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Display the plot
plt.show()