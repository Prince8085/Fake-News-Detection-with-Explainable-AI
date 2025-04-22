import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load and train model (reuse code from main.py) ---
@st.cache_resource

def load_model_and_vectorizer():
    fake_path = '../News_dataset/Fake.csv'
    true_path = '../News_dataset/True.csv'
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = df['text'] if 'text' in df.columns else df.iloc[:,0]
    y = df['label']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)
    # Calculate metrics
    y_pred = model.predict(X_vec)
    report = classification_report(y, y_pred, output_dict=True)
    accuracy = accuracy_score(y, y_pred)
    return model, vectorizer, report, accuracy, df

model, vectorizer, report, accuracy, df = load_model_and_vectorizer()

# --- Streamlit UI ---
st.title('Fake News Detection with Explainable AI')
st.write('Enter a news article below to check if it is Fake or Real. The app will also explain the prediction!')

# Show model metrics
with st.expander('Show Model Performance Metrics'):
    st.write(f"**Accuracy:** {accuracy:.3f}")
    st.write(pd.DataFrame(report).transpose())

# Example articles
st.markdown('---')
st.subheader('Try an Example Article')
example_idx = st.selectbox('Select example:', options=list(range(5)), format_func=lambda i: f"Example {i+1}")
st.write(df.iloc[example_idx]['text'][:500] + '...')
if st.button('Use this example'):
    st.session_state['user_input'] = df.iloc[example_idx]['text']

# User input
user_input = st.text_area('News Article Text', value=st.session_state.get('user_input', ''), height=200)

# File upload
st.markdown('---')
st.subheader('Batch Prediction (Upload CSV)')
uploaded_file = st.file_uploader('Upload a CSV file with a column named "text"', type=['csv'])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    if 'text' not in batch_df.columns:
        st.error('CSV must have a column named "text".')
    else:
        batch_pred = model.predict(vectorizer.transform(batch_df['text']))
        batch_proba = model.predict_proba(vectorizer.transform(batch_df['text']))
        batch_df['prediction'] = batch_pred
        batch_df['confidence'] = batch_proba.max(axis=1)
        batch_df['prediction_label'] = batch_df['prediction'].map({0: 'Fake', 1: 'Real'})
        st.write(batch_df[['text', 'prediction_label', 'confidence']])
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Results as CSV', csv, 'predictions.csv', 'text/csv')

# Single prediction and explanation
if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter some text.')
    else:
        X_input = [user_input]
        pred_proba = model.predict_proba(vectorizer.transform(X_input))[0]
        pred_label = np.argmax(pred_proba)
        label_map = {0: 'Fake', 1: 'Real'}
        st.markdown(f'**Prediction:** {label_map[pred_label]} (Confidence: {pred_proba[pred_label]:.2f})')

        # LIME explanation
        explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])
        def predict_proba(texts):
            return model.predict_proba(vectorizer.transform(texts))
        exp = explainer.explain_instance(user_input, predict_proba, num_features=10)
        exp_list = exp.as_list()
        st.markdown('**Top words influencing the prediction:**')
        # Bar chart for explanation
        words, weights = zip(*exp_list)
        fig, ax = plt.subplots()
        sns.barplot(x=list(weights), y=list(words), orient='h', ax=ax, palette='coolwarm')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Word')
        st.pyplot(fig)
