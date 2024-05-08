from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

def load_model_and_resources(svm_model_path, tokenizer_path, tfidf_path, encoder_path):
    svm_model = None
    tokenizer = None
    tfidf = None
    label_encoder = None
    
    try:
        with open(svm_model_path, 'rb') as svm_file:
            svm_model = pickle.load(svm_file)
            logging.info("SVM model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load SVM model: {e}")

    try:
        with open(tokenizer_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
            logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")

    try:
        with open(tfidf_path, 'rb') as tfidf_file:
            tfidf = pickle.load(tfidf_file)
            logging.info("TF-IDF vectorizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load TF-IDF vectorizer: {e}")

    try:
        with open(encoder_path, 'rb') as encoder_file:
            label_encoder = pickle.load(encoder_file)
            logging.info("Label encoder loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load label encoder: {e}")

    if not all([svm_model, tokenizer, tfidf, label_encoder]):
        logging.error("Not all resources were loaded successfully.")
        raise ValueError("Model and resources could not be loaded.")

    return svm_model, tokenizer, tfidf, label_encoder

# Place your paths here
svm_model_path = './student_advisory_svm_model.sav'
tokenizer_path = './tokenizer.pickle'
tfidf_path = './tfidf.pickle'
encoder_path = './label_encoder.pickle'

# Call the function directly for testing purposes
svm_model, tokenizer, tfidf, label_encoder = load_model_and_resources(
    svm_model_path, tokenizer_path, tfidf_path, encoder_path
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the question from the request
        data = request.get_json(force=True)
        question = data['question']
        
        # Check if tfidf is loaded
        if not tfidf:
            raise ValueError("TF-IDF vectorizer is not loaded.")
            
        # Preprocess the question for SVM
        tfidf_vector = tfidf.transform([question]).toarray()

        # Check if SVM model is loaded
        if not svm_model:
            raise ValueError("SVM model is not loaded.")
            
        # Predict using SVM model
        predicted_category_index = svm_model.predict(tfidf_vector)[0]
        predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]

        logging.info(f"Response: {predicted_category}")

        # Return the predicted category
        return jsonify(predicted_category=predicted_category)
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)