# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from pathlib import Path
import shutil
import zipfile
import base64

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('mini_project.h5')

# Function to make predictions on the input images
def classify_image(image, model):
    image = np.array(image.resize((224, 224)))  # Resize the image to match your model's input shape
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    return prediction

# Function to create a separate folder for classified notes images
def create_notes_folder():
    notes_folder = 'classified_notes'
    if not os.path.exists(notes_folder):
        os.mkdir(notes_folder)
    return notes_folder

# Function to create a zip file of the classified notes folder
def create_zip_file(folder_path, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

# Streamlit app
def main():
    st.title('Classify Classroom Notes')

    # Load the model
    model = load_model()

    # Get folder path from the user
    folder_path = st.text_input('Enter the path of the folder containing images:')
    folder_path = Path(folder_path)

    if not folder_path.exists():
        st.write('Invalid folder path.')
        return

    notes_folder = create_notes_folder()

    # Process all images in the folder
    for image_path in folder_path.glob('*.jpg'):  # You can add other supported extensions if needed
        image = Image.open(image_path)
        prediction = classify_image(image, model)
        # class_label = np.argmax(prediction)  # Get the index of the highest predicted class
        class_label = prediction * 10

        if class_label > 1:  # 1 represents the "notes" class
            # st.write("This is a notes image.")
            shutil.move(str(image_path), os.path.join(notes_folder, image_path.name))
        # else:
        #     st.write("This is not a notes image.")

    st.write("Classification completed. Notes images are saved in the 'classified_notes' folder.")

    # Provide a link to download the classified_notes folder as a zip file
    if os.path.exists(notes_folder) and len(os.listdir(notes_folder)) > 0:
        zip_filename = 'classified_notes.zip'
        create_zip_file(notes_folder, zip_filename)
        st.write("### Download Classified Images ###")
        st.markdown(get_binary_file_downloader_html(zip_filename, 'Download Classified Images'), unsafe_allow_html=True)

# Function to create a link to download a file
def get_binary_file_downloader_html(bin_file, label='Download'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{label}</a>'
    return href

if __name__ == '__main__':
    main()
