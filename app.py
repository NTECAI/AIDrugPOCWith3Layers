import os
import io
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from ultralytics import YOLO
from PIL import Image, ImageDraw
import json

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import wget
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


# Function to load a specified model
def load_model(model_name):
    model = YOLO(f'models/{model_name}')
    return model

# Load shape identification model
shape_model = load_model('shape_best_v8.pt')

# # Load color identification model
color_model = load_model('color_best_v8.pt')

def infer_result(model, image):
    res = model(image)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted)
    try:
        return json.loads(res[0].tojson())[0]['name']
    except:
        return ""

# Streamlit app
def main():
    st.title("Object Identification")

    option = st.selectbox("Options", ("Upload Image", "Capture Image"))

    if option == "Capture Image":
        st.write("Click the button below to capture an image:")
        picture = st.camera_input("Take a picture")
        uploaded_file = picture

    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    

    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)

        st.header("1. Color:") 
        color = infer_result(color_model, image)
        st.write(color)

        st.header("2. Shape:") 
        shape = infer_result(shape_model, image)
        shape

        if uploaded_file.size > 4 * 1024 * 1024:  # more than 4MB
            st.write("Resizing image...")
            image = resize_image(image)

        st.header("3. OCR results:")    
        OCR_layer(image, color, shape)

def resize_image(image):
    # Get current dimensions
    width, height = image.size
    
    # Calculate new dimensions
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    st.write(resized_image.size)
    return resized_image

def get_OCR_results(image):
    
    endpoint = "https://jiminstance3.cognitiveservices.azure.com/"
    key = "5d5d09b9bb0a49cbb22a0ade99f0f2ca"

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    if image.format is None:
        image_format = 'JPEG'
    else:
        image_format = image.format
    image.save(img_byte_arr, format=image_format)
    img_byte_arr = img_byte_arr.getvalue()

    # Polling the service for the result
    poller = document_analysis_client.begin_analyze_document("prebuilt-document", img_byte_arr)
    result = poller.result()

    # Extract and display the information
    extracted_text  = [line.content for page in result.pages for line in page.lines]
    return extracted_text 

def OCR_layer(image, color, shape):

    # data preprocessing
    df = pd.read_csv("csv_Drugimg_AHNH_archive.csv",encoding="Big5")
    df = df[["code","color","shape","imprint1", "imprint2"]][df["imprint1"].notna()]
    df['shape'] = df['shape'].str.replace(' ', '')
    # st.dataframe(df) 

    extracted_text = get_OCR_results(image)
    extracted_text
    # Filter results


    # filter imprints
    if not extracted_text:
        possible_drugs = df[(df['imprint1']=='[NOIMPRINT]') | (df['imprint2']=='[NOIMPRINT]')]
    else:
        search_terms = '|'.join(extracted_text)
        possible_drugs = df[(df['imprint1'].str.contains(search_terms)) | (df['imprint2'].str.contains(search_terms))]

    st.write("Filter: OCR only")
    st.dataframe(possible_drugs) 

    # filter color
    possible_drugs = possible_drugs[possible_drugs['color'].str.contains(color)]

    # filter shape
    possible_drugs = possible_drugs[possible_drugs['shape'].str.contains(shape)]

    st.write("Filter: OCR + color + shape")
    st.dataframe(possible_drugs) 


if __name__ == "__main__":
    main()

