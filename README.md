# EchoGrade

# Grammar Scoring Engine for Voice Samples

This project provides a web interface for a Grammar Scoring Engine that analyzes voice samples and provides a grammar proficiency score.

## Features

- Record audio directly from the browser
- Upload existing audio files
- AI-powered grammar analysis
- Visual score display with feedback
- Responsive design for mobile and desktop

## Project Architecture

The project consists of two main components:

1. **Frontend**: React application with TypeScript and Tailwind CSS
2. **Backend**: Flask API serving the ML model for grammar scoring

## Setup Instructions

### Prerequisites

- Node.js (v14+)
- Python (v3.7+)
- pip (Python package manager)

### Frontend Setup

```sh
# Clone the repository
git clone <YOUR_GIT_URL>

# Navigate to the project directory
cd <YOUR_PROJECT_NAME>

# Install dependencies
npm i

# Start the development server
npm run dev
```

### Backend Setup

```sh
# Create a model directory
mkdir -p model

# Copy your trained model to the model directory
cp grammar_scoring_model.joblib model/

# Navigate to the backend directory
cd src/backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install flask flask-cors numpy librosa scikit-learn joblib

# Start the Flask server
python app.py
```

## Deployment

### Frontend Deployment

The React frontend can be deployed to Vercel, Netlify, or any static hosting service.

```sh
# Build the production version
npm run build

# Deploy to your chosen platform
# For example, with Vercel:
vercel deploy
```

### Backend Deployment

The Flask backend can be deployed to platforms like Heroku, AWS, or Google Cloud.

```sh
# Create requirements.txt
pip freeze > requirements.txt

# Deploy to your chosen platform
# For Heroku example:
heroku create
git push heroku main
```

## Machine Learning Model

The grammar scoring engine uses a RandomForestRegressor model trained on labeled audio samples. The model extracts audio features including MFCCs and chroma features to predict grammar scores on a scale from 0 to 10.

### Feature Extraction

- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma features
- Statistical measures (mean, standard deviation)

## Integration with Your Existing Jupyter Model

The web application is designed to work with your existing trained model saved as `grammar_scoring_model.joblib`. The model should be placed in the `model` directory for the Flask API to load it.

## License

[MIT License](LICENSE)
