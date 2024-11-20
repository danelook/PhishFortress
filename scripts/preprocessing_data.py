import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import joblib

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# List of datasets to load (fixed missing commas)
datasets = [
    r'C:\Users\novem\phishfortress\data\CEAS_08.csv',
    r'C:\Users\novem\phishfortress\data\Nazario.csv',
    r'C:\Users\novem\phishfortress\data\Nigerian_Fraud.csv',
    r'C:\Users\novem\phishfortress\data\SpamAssasin.csv',
    r'C:\Users\novem\phishfortress\data\Ling.csv',
    r'C:\Users\novem\phishfortress\data\phishing_email.csv',
    r'C:\Users\novem\phishfortress\data\Enron.csv',
    r'C:\Users\novem\phishfortress\data\emails1.csv',
    r'C:\Users\novem\phishfortress\data\emails2.csv',
    r'C:\Users\novem\phishfortress\data\emails3.csv'
]

# Load and combine all datasets into one DataFrame
df_list = []
for filepath in datasets:
    try:
        # Ensure individual file is loaded correctly
        print(f"Loading file: {filepath}")
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
        # Ensure the necessary columns are present
        if 'subject' in df.columns and 'body' in df.columns and 'label' in df.columns:
            df['email_body'] = df['subject'] + ' ' + df['body']
            df_list.append(df[['email_body', 'label']])
        elif 'Email Text' in df.columns and 'Email Type' in df.columns:
            df.rename(columns={'Email Text': 'email_body', 'Email Type': 'label'}, inplace=True)
            df_list.append(df[['email_body', 'label']])
        else:
            print(f"Required columns not found in {filepath}. Skipping this dataset.")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

# Check if there are any DataFrames to concatenate
if not df_list:
    raise ValueError("No valid datasets were loaded. Please check your files and try again.")

# Combine all DataFrames
df = pd.concat(df_list, ignore_index=True)

# Drop rows with missing values in 'email_body' or 'label'
df.dropna(subset=['email_body', 'label'], inplace=True)

print("Dataset loaded successfully.")

# Function to clean HTML tags
def clean_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text()

# Function to remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

print("Starting preprocessing...")

# Preprocess the email body
df['email_body'] = df['email_body'].apply(clean_html)  # Remove HTML
df['email_body'] = df['email_body'].str.lower()  # Lowercase
df['email_body'] = df['email_body'].apply(remove_punctuation)  # Remove punctuation
df['email_body'] = df['email_body'].replace(r'\d+', '', regex=True)  # Remove numbers
df['email_body'] = df['email_body'].apply(remove_stop_words)  # Remove stop words

print("Preprocessing completed.")

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['email_body'])

# Splitting the dataset
from sklearn.model_selection import train_test_split

y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Save the vectorizer and datasets for future use
model_dir = r'C:\Users\novem\phishfortress\models'
os.makedirs(model_dir, exist_ok=True)

joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
joblib.dump((X_train, y_train), os.path.join(model_dir, 'train_data.pkl'))
joblib.dump((X_test, y_test), os.path.join(model_dir, 'test_data.pkl'))

print("Data saved successfully.")

