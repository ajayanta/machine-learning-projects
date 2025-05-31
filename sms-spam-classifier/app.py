import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


# Initialize the stemmer
ps = PorterStemmer()

# Set the Streamlit page configuration
st.set_page_config(page_title="Sms spam classifier")  # optional fix typo here too

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # remove language='english' argument
    y = []

    # Keep only alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y
    y = []

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model from pickle files
with open("vectorizer.pkl", 'rb') as f:
    cv = pickle.load(f)

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("SMS/Email Spam Classifier")
input_text = st.text_area("Enter sms/email here")

# Predict button
if st.button('predict'):
    transformed_text = transform_text(input_text)
    vector_input = cv.transform([transformed_text])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
