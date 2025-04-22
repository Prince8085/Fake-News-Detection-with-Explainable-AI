Fake News Detection with Explainable AI
Overview
This project uses machine learning and explainable AI to classify news articles as Fake or Real. It uses a Logistic Regression model and LIME for model interpretability.

Dataset
Fake.csv and True.csv from the News_dataset folder.
Features
Data preprocessing and vectorization
Model training and evaluation
LIME-based explanation for predictions
How to Run
Install requirements:
pip install -r requirements.txt
Make sure the dataset folder is at ../News_dataset/ relative to the script.
Run the main script:
python main.py
To launch the web app:
streamlit run app.py
Project Structure
main.py: Main script for training, evaluation, and explanation
app.py: Streamlit web app for interactive predictions and explanations
requirements.txt: List of dependencies
../News_dataset/: Folder containing Fake.csv and True.csv
Example Output
Classification report (accuracy, precision, recall)
LIME explanation for a sample prediction
Author
Prince Kachhwaha
