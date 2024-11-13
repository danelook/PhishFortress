import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# List of datasets to load, excluding emails1.csv, TREC_05.csv, and TREC_06.csv
datasets = [
    r'C:\Users\novem\phishfortress\data\emails2.csv',
    r'C:\Users\novem\phishfortress\data\emails3.csv',
    r'C:\Users\novem\phishfortress\data\TREC_07.csv',
    r'C:\Users\novem\phishfortress\data\Enron (1).csv',
    r'C:\Users\novem\phishfortress\data\Nazario.csv',
    r'C:\Users\novem\phishfortress\data\Ling.csv',
    r'C:\Users\novem\phishfortress\data\SpamAssasin.csv',
    r'C:\Users\novem\phishfortress\data\Nazario_5.csv',
    r'C:\Users\novem\phishfortress\data\Nigerian_Fraud.csv'
]

# Load and combine all datasets into one DataFrame
df_list = []

for filepath in datasets:
    try:
        # Load with error handling for malformed lines
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
        
        # Standardize column names if needed
        if 'subject' in df.columns and 'body' in df.columns and 'label' in df.columns:
            # Combine subject and body into a single field for other datasets
            df['email_text'] = df['subject'] + ' ' + df['body']
        elif 'Email Text' in df.columns and 'Email Type' in df.columns:
            # Handling for emails1.csv-style format (if similar datasets are found in future)
            df.rename(columns={'Email Text': 'email_text', 'Email Type': 'label'}, inplace=True)
        else:
            print(f"Required columns not found in {filepath}. Skipping this dataset.")
            continue

        # Drop unnecessary columns and keep only required ones
        if 'email_text' in df.columns and 'label' in df.columns:
            df = df[['email_text', 'label']]
            
            # Ensure labels are discrete and standardized (e.g., binary values)
            df['label'] = df['label'].astype(str).str.lower().map({
                'spam': 1, 'ham': 0, 'phish': 1, 'legit': 0, '0': 0, '1': 1, 'safe email': 0, 'phishing email': 1
            })
            
            # Drop rows with invalid labels
            df.dropna(subset=['label'], inplace=True)
            df['label'] = df['label'].astype(int)

            # Add to list of DataFrames
            df_list.append(df)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

# Check if there are any DataFrames to concatenate
if not df_list:
    raise ValueError("No valid datasets were loaded. Please check your files and try again.")

# Combine all DataFrames
df = pd.concat(df_list, ignore_index=True)

# Drop rows with missing values in 'email_text' or 'label'
df.dropna(subset=['email_text', 'label'], inplace=True)

# Splitting the dataset into features and labels
X = df['email_text']
y = df['label']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the email text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a basic Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Create a folder for models if it doesn't exist
model_dir = r'C:\Users\novem\phishfortress\models'
os.makedirs(model_dir, exist_ok=True)

# Save the models and vectorizer under new names
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


