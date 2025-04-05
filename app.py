import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model - using the path from your project structure
MODEL_PATH = "grammar_scoring_model.joblib"  # Model is in the root directory
model = None

def load_model():
    """
    Loads the trained grammar scoring model.
    """
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")

def extract_features(audio_path, sr=22050, n_mfcc=13, duration=5):
    """
    Extracts audio features from an audio file.
    
    Parameters:
      audio_path (str): Path to the audio file.
      sr (int): Sampling rate.
      n_mfcc (int): Number of MFCCs to extract.
      duration (float): Duration (in seconds) to consider.
    
    Returns:
      np.array: Feature vector containing mean and std of MFCC and chroma features.
    """
    try:
        # Load audio file
        audio_data, sr = librosa.load(audio_path, sr=sr)
        
        # Ensure we only use a defined duration for consistency
        if len(audio_data) > duration * sr:
            audio_data = audio_data[:int(duration * sr)]
        
        # Extract MFCC features and compute statistics
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Extract Chroma features and compute statistics
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Combine features into a single vector
        features = np.concatenate((mfccs_mean, mfccs_std, chroma_mean, chroma_std))
        return features
    except Exception as e:
        print(f"Error processing audio file: {e}")
        # Create a zero vector in case of error
        return np.zeros(n_mfcc*2 + 24)

@app.route("/")
def home():
    """
    Home route for the API.
    """
    return (
        "<h1>Welcome to the Grammar Scoring API!</h1>"
        "<p>Use the <code>/api/score</code> endpoint to upload an audio file and get a grammar score.</p>"
        "<p>Check <code>/api/health</code> for health status of the API.</p>"
    )

@app.route("/api/score", methods=["POST"])
def score_audio():
    """
    Endpoint to score an audio file for grammar.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    try:
        # Get the audio file
        audio_file = request.files["audio"]
        
        # Save to a temporary file
        temp_path = "temp_audio.wav"
        audio_file.save(temp_path)
        
        # Extract features
        features = extract_features(temp_path)
        
        # Make prediction
        score = float(model.predict([features])[0])
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            "score": score,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Endpoint to check health status of the API.
    """
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
