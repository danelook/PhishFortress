from flask import Flask, request, render_template, jsonify
import joblib
import os

app = Flask(__name__)

# Get the absolute path for the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'models/phishing_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models/vectorizer.pkl')

# Load the saved model and vectorizer using the absolute path
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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

    # Render the classification result in the same HTML page
    return render_template('index.html', classification=result)

if __name__ == '__main__':
    app.run(debug=True)