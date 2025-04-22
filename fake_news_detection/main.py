import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import lime
import lime.lime_text
import numpy as np
import os

# Load data
def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0  # Fake
    true_df['label'] = 1  # Real
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    return df

# Preprocess data
def preprocess(df):
    X = df['text'] if 'text' in df.columns else df.iloc[:,0]
    y = df['label']
    return X, y

def main():
    # Paths to data
    fake_path = '../News_dataset/Fake.csv'
    true_path = '../News_dataset/True.csv'

    # Load and preprocess
    df = load_data(fake_path, true_path)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    # Explain with LIME
    class_names = ['Fake', 'Real']
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    idx = 0  # Explain the first test sample
    def predict_proba(texts):
        return model.predict_proba(vectorizer.transform(texts))
    exp = explainer.explain_instance(X_test.iloc[idx], predict_proba, num_features=10)
    print(f'\nLIME explanation for sample {idx}:')
    print(exp.as_list())

if __name__ == '__main__':
    main()
