import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import zipfile
import os

#
#
# Path to your zip file (relative path)
zip_file_path = 'Male vs Female trained model_50 tt_epochs VGG  part 2 till now best.zip'

# Directory where you want to extract the contents of the zip file (relative path)
extract_dir = 'unzipped_model'

# Get the absolute path of the current directory
current_directory = os.getcwd()

# Combine the current directory with the relative paths
zip_file_path = os.path.join(current_directory, zip_file_path)
extract_dir = os.path.join(current_directory, extract_dir)

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# List the files in the extraction directory
extracted_files = os.listdir(extract_dir)
print("Extracted files:", extracted_files)

# Now you can load your model from the extracted files
# For example, if your model file is an h5 file, you can load it like this:
model_filename = [f for f in extracted_files if f.endswith('.h5')][0]
model_path = os.path.join(extract_dir, model_filename)
model = tf.keras.models.load_model(model_path)



# # # Load your trained models
# model_path = r'D:\Venkatesh\Projects\CNN Projects\Gender classification project\Male vs Female trained model_50 tt_epochs VGG  part 2 till now best.h5'
# # model_path = 'Male vs Female trained model_50 tt_epochs VGG  part 2 till now best.h5'
# model = tf.keras.models.load_model(model_path)


# Define function to make predictions
def predict_gender(image):
    # Resize the image
    resized_image = image.resize((256, 256))

    # Convert to numpy array
    img_array = np.array(resized_image)

    # Reshape for model input
    test_input = np.expand_dims(img_array, axis=0)

    # Predict gender
    prediction = model.predict(test_input)

    # Assuming 'prediction' contains the probability value
    probability = prediction[0][0]

    # Apply a threshold (0.5 in this case) to classify as 0 or 1
    if probability >= 0.5:
        return "Male"
    else:
        return "Female"


# Define the Streamlit UI
st.title('⚥ GenderAI Classifier:  Empowering Insights')
st.markdown("*by Venkatesh*")

print("\n")
print("")
print("")
print("")

st.write("""



Welcome to the **GenderAI Classifier**! 🎉

Discover the power of artificial intelligence with this application, which leverages state-of-the-art deep learning techniques to predict gender. Simply upload an image, and let the CNN model do the rest.
""")


# # Set background image using an online URL
# page_bg_img = '''
# <style>
#   body {
#     background-image: url("https://raw.githubusercontent.com/Jinkyiyer/Gender-classification/main/ai%20background.jpg");
#     background-size: cover;
#   }
# </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)


# Set background image using an online URL
page_bg_img = '''
<style>
  body {
    background-image: url("https://raw.githubusercontent.com/Jinkyiyer/Gender-classification/ai background.jpg");
    background-size: cover;
  }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# Example URL to hero image hosted on GitHub
hero_img_url = "https://raw.githubusercontent.com/Jinkyiyer/Gender-classification/main/ai gif heroimg.webp"
# Display hero image
st.image(hero_img_url, use_column_width=True)


# Add text instructions
st.write("Please upload a photo similar to a passport-size photo for better classification results.")
st.write("Here are some examples of suitable photos:")

# Display example photos
# example_photo_paths = ["example1.jpg", "example2.jpeg", "example3.jpeg"]  # Paths to your example photos
# for example_photo_path in example_photo_paths:
#     example_image = Image.open(example_photo_path)
#     st.image(example_image, caption='Example Photo', use_column_width=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when 'Predict' button is clicked
    if st.button('Predict'):
        prediction = predict_gender(image)
        st.title('Prediction: {}'.format(prediction))


import streamlit as st

image_path = "example1.jpg"
print(f"Using image path: {image_path}")

page_bg_img = '''
<style>
  body {
    background-image: url("%s");
    background-size: cover;
  }
</style>
''' % image_path
st.markdown(page_bg_img, unsafe_allow_html=True)


