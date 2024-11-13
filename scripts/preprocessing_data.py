# Updated preprocessing_data.py
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import joblib

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# List of datasets to load
datasets = [
    r'C:\Users\novem\phishfortress\data\emails2.csv',
    r'C:\Users\novem\phishfortress\data\emails3.csv',
    r'C:\Users\novem\phishfortress\data\TREC_07.csv',
    r'C:\Users\novem\phishfortress\data\Enron.csv',
    r'C:\Users\novem\phishfortress\data\Nazario.csv',
    r'C:\Users\novem\phishfortress\data\Ling.csv',
    r'C:\Users\novem\phishfortress\data\SpamAssasin.csv',
    r'C:\Users\novem\phishfortress\data\Nigerian_Fraud.csv'
]

# Load and combine all datasets into one DataFrame
df_list = []

for filepath in datasets:
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
        
        # Standardize columns if needed
        if 'subject' in df.columns and 'body' in df.columns and 'label' in df.columns:
            df['email_body'] = df['subject'] + ' ' + df['body']
        elif 'Email Text' in df.columns and 'Email Type' in df.columns:
            df.rename(columns={'Email Text': 'email_body', 'Email Type': 'label'}, inplace=True)
        else:
            print(f"Required columns not found in {filepath}. Skipping this dataset.")
            continue
        
        # Keep only required columns
        if 'email_body' in df.columns and 'label' in df.columns:
            df = df[['email_body', 'label']]
            df_list.append(df)

    except Exception as e:
        print(f"Error loading {filepath}: {e}")

# Combine all DataFrames
df = pd.concat(df_list, ignore_index=True)

# Preprocess the email body
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

print("Starting preprocessing...")
df['email_body'] = df['email_body'].apply(clean_html)
df['email_body'] = df['email_body'].str.lower()
df['email_body'] = df['email_body'].apply(remove_punctuation)
df['email_body'] = df['email_body'].replace(r'\d+', '', regex=True)
df['email_body'] = df['email_body'].apply(remove_stop_words)
print("Preprocessing completed.")

# Save the cleaned dataset for training
df.to_csv(r'C:\Users\novem\phishfortress\data\preprocessed_emails.csv', index=False)
