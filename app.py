import os
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

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
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'models/{model_name}')
    model.eval()
    return model

# Load shape identification model
shape_model = load_model('shape_best.pt')

# # Load color identification model
color_model = load_model('color_best.pt')

# Define the list of models
models = {
    # ('Green', 'Round'): ['green_round_best.pt'],
    # ('Orange', 'Capsule'): ['orange_capsule_best.pt'],
    ('Orange', 'Round'): ['orange_round_1_best.pt', 'orange_round_2_best.pt'],
    ('White', 'Capsule'): ['white_capsule_best.pt'],
    ('White', 'Round'): ['white_round_1_best.pt', 'white_round_6_best.pt'],
}

# Function to identify objects based on a given model
def identify_objects(image, model):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run object identification
    results = model(image)

    # Process results
    objects = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        name = model.names[int(cls)]
        if conf.item() >0.6:
            objects.append([name, conf.item(), (x1, y1, x2, y2)])

    return objects


def add_background(image):
    # Create a new image with the desired background color
    new_image = Image.new("RGB", image.size, "black")

    # Paste the original image onto the new image
    new_image.paste(image, (0, 0))

    return new_image

def draw_bounding_boxes(image, objects):

    image = add_background(image)

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Create a copy of the image for drawing bounding boxes
    output_image = image_array.copy()

    # Draw bounding boxes on the output image
    for obj in objects:
        name, bbox = obj[0], obj[1]
        x1, y1, x2, y2 = bbox

        cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
        cv2.putText(output_image, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # Convert the output image back to PIL format
    output_image = Image.fromarray(output_image)

    return output_image

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

    

        # Perform shape identification
        shapes = identify_objects(image, shape_model)

        st.header("1. Shapes:")
        for shape in shapes[:1]:
            st.write(shape[:2])


        # Perform color identification
        colors = identify_objects(image, color_model)

        st.header("2. Colors:")
        for color in colors[:1]:
            st.write(color[:2])

        # # Call the specified model based on color and shape combination
        # color_label = colors[0][0] if colors else None
        # shape_label = shapes[0][0] if shapes else None
        
    
        # if color_label and shape_label and (color_label, shape_label) in models:
        #     model_names = models[(color_label, shape_label)]
        #     st.header("3. Identified Drugs:")
        #     for model_name in model_names:
        #         loaded_model = load_model(model_name)
        #         drugs = identify_objects(image, loaded_model)
        #         # threshold control
        #         if drugs:
        #             st.write(model_name, drugs[:1][0][:2])

        #             # Prepare objects for drawing bounding boxes
        #             objects = [(drug[0], drug[2]) for drug in drugs]
        #             # st.write(objects)
        #             # Draw bounding boxes on the image
        #             output_image = draw_bounding_boxes(image, objects)

        #             # Display the image with bounding boxes
        #             st.image(output_image, channels="RGB")
        # else:
        #     st.error("No model found.")

        
        st.header("3. OCR results:")    
        OCR_layer(uploaded_file)
    


def OCR_layer(uploaded_file):
    df = pd.read_csv("csv_Drugimg_AHNH_archive.csv",encoding="Big5")
    df = df[["code","color","shape","imprint1", "imprint2"]][df["imprint1"].notna()]
    # st.dataframe(df) 

    # Replace 'your-endpoint' and 'your-key' with your Azure endpoint and key
    endpoint = "https://jiminstance123.cognitiveservices.azure.com/"
    key = "38b8408fcb7e42488b8a4c2289997a07"

    # Create a client to interact with the Azure Form Recognizer service
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    # Polling the service for the result
    poller = document_analysis_client.begin_analyze_document("prebuilt-document", file_bytes)
    result = poller.result()

    # Extract and display the information
    extracted_text  = [line.content for page in result.pages for line in page.lines]
    extracted_text 

    if not extracted_text:
        possible_drugs = df[(df['imprint1']=='[NOIMPRINT]') | (df['imprint2']=='[NOIMPRINT]')]
    else:
        search_terms = '|'.join(extracted_text)
        possible_drugs = df[(df['imprint1'].str.contains(search_terms)) | (df['imprint2'].str.contains(search_terms))]

    st.dataframe(possible_drugs) 


if __name__ == "__main__":
    main()

