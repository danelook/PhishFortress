import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
import joblib
import os

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# List of datasets to preprocess
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

# Load and preprocess each dataset
df_list = []

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

for filepath in datasets:
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')

        # Standardize column names for consistency
        if 'subject' in df.columns and 'body' in df.columns and 'label' in df.columns:
            df['email_text'] = df['subject'] + ' ' + df['body']
        elif 'Email Text' in df.columns and 'Email Type' in df.columns:
            df.rename(columns={'Email Text': 'email_text', 'Email Type': 'label'}, inplace=True)
        else:
            print(f"Required columns not found in {filepath}. Skipping this dataset.")
            continue

        # Drop unnecessary columns and keep only required ones
        if 'email_text' in df.columns and 'label' in df.columns:
            df = df[['email_text', 'label']]

            # Apply preprocessing on email_text
            df['email_text'] = df['email_text'].apply(preprocess_text)

            # Ensure labels are standardized
            df['label'] = df['label'].astype(str).str.lower().map({
                'spam': 1, 'ham': 0, 'phish': 1, 'legit': 0, '0': 0, '1': 1, 'safe email': 0, 'phishing email': 1
            })

            # Drop rows with invalid labels
            df.dropna(subset=['label'], inplace=True)
            df['label'] = df['label'].astype(int)

            # Add the preprocessed DataFrame to the list
            df_list.append(df)
        else:
            print(f"Required columns not found after processing in {filepath}. Skipping this dataset.")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

# Check if there are any DataFrames to concatenate
if not df_list:
    raise ValueError("No valid datasets were loaded. Please check your files and try again.")

# Combine all DataFrames
df = pd.concat(df_list, ignore_index=True)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['email_text'])

# Splitting the dataset
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Create a folder for models if it doesn't exist
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Save the vectorizer and datasets for future use
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
joblib.dump((X_train, y_train), os.path.join(model_dir, 'train_data.pkl'))
joblib.dump((X_test, y_test), os.path.join(model_dir, 'test_data.pkl'))

print("Data saved successfully.")
