# Fake News Detection with Explainable AI

## Overview
This project uses machine learning and explainable AI to classify news articles as Fake or Real. It uses a Logistic Regression model and LIME for model interpretability.

## Dataset
- `Fake.csv` and `True.csv` from the `News_dataset` folder.

## Features
- Data preprocessing and vectorization
- Model training and evaluation
- LIME-based explanation for predictions

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure the dataset folder is at `../News_dataset/` relative to the script.
3. Run the main script:
   ```bash
   python main.py
   ```
4. To launch the web app:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `main.py`: Main script for training, evaluation, and explanation
- `app.py`: Streamlit web app for interactive predictions and explanations
- `requirements.txt`: List of dependencies
- `../News_dataset/`: Folder containing `Fake.csv` and `True.csv`

## Example Output
- Classification report (accuracy, precision, recall)
- LIME explanation for a sample prediction

## Author
Your Name Here
