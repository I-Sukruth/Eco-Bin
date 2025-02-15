import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import gdown  # To download from Google Drive
from PIL import Image

# Download the model from Google Drive
@st.cache_resource
def load_model_from_drive():
    url = 'https://drive.google.com/file/d/1ucpbyJY61OnCSRKbrj4FgYBSXX8FSONy/view?usp=sharing' 
    output = 'Eco_Bin.h5'
    gdown.download(url, output, quiet=False)
    return load_model(output)

# Load the model
model = load_model_from_drive()

# Define the class labels (according to your folder structure)
class_labels = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# Function to make predictions
def predict_image(img):
    img = img.resize((128, 128))  # Resize the image to match the input size of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class_index]

# Streamlit app layout
st.title('Eco-Bin')

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image of the material", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    uploaded_img = Image.open(uploaded_file)
    st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)

    # Make prediction when the user clicks the button
    if st.button('Predict'):
        with st.spinner('Classifying...'):
            prediction = predict_image(uploaded_img)
            st.success(f'The Predicted Category is: {prediction}')
