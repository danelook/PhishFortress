import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
import joblib

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('C:\\User\\novem\\phishfortress\\data\\emails.csv')
print("Dataset loaded successfully.")

# Function to clean HTML tags
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

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
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['email_body'])

# Splitting the dataset
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Save the vectorizer and datasets for future use
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump((X_train, y_train), 'models/train_data.pkl')
joblib.dump((X_test, y_test), 'models/test_data.pkl')

print("Data saved successfully.")

