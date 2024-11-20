from flask import Flask, request, render_template, jsonify
import joblib
import os
import csv
from cryptography.fernet import Fernet

app = Flask(__name__)

# Paths for model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'models/phishing_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models/vectorizer.pkl')

# Load the saved model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Load the encryption key
key_path = os.path.join(os.path.dirname(__file__), 'secrets', 'secret.key')
with open(key_path, 'rb') as key_file:
    key = key_file.read()
cipher = Fernet(key)

# Path for the CSV file to store classified emails
csv_path = os.path.join(os.path.dirname(__file__), 'classified_emails.csv')

# Ensure the CSV file has a header row if it doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Encrypted Email Text', 'Classification'])

# Function to save and encrypt email text with classification
def save_classification(email_text, classification):
    # Encrypt the email text
    encrypted_text = cipher.encrypt(email_text.encode())
    
    # Save the encrypted email and classification to the CSV file
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([encrypted_text.decode(), classification])

# Serve the HTML page for user input
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission from the HTML page
@app.route('/classify', methods=['POST'])
def classify():
    # Get the email text from the form
    email_text = request.form['email_text']
    
    if not email_text:
        return render_template('index.html', classification='No email text provided')

    # Preprocess and vectorize the input email text
    email_features = vectorizer.transform([email_text])

    # Use the model to predict if it's phishing or legitimate
    prediction = model.predict(email_features)[0]
    result = 'Phishing' if prediction == 1 else 'Legitimate'

    # Save the encrypted classification result
    save_classification(email_text, result)

    # Render the classification result in the same HTML page
    return render_template('index.html', classification=result)

if __name__ == '__main__':
    app.run(debug=True)