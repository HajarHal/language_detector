import streamlit as st
import pickle

# Load the trained model
with open('lrmodel.pckl', 'rb') as lrfile:
    pipe_lr = pickle.load(lrfile)

# Title and description in the main section
st.markdown('<h1 style="color: #017bff;">Language Identification Application</h1>', unsafe_allow_html=True)
st.write("Welcome to the Language Identification Model application.")

# Sidebar with logo and description
st.sidebar.image("lo.png", width=10 ,use_column_width=True)
st.sidebar.title("About")
st.sidebar.info(
    """
    This application employs state-of-the-art machine learning methodologies to predict the language of text inputs. By harnessing the power of Logistic Regression, it achieves exceptional accuracy in identifying languages, making it a valuable tool for various language-related tasks.
    """
)

# Input text box
input_text = st.text_area("Enter text here:")

# Predict button
if st.button("Predict"):
    if input_text:
        prediction = pipe_lr.predict([input_text])[0]
        st.write(f"The predicted language is: <span style='color:green'><b>{prediction}</b></span>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text for prediction.")
