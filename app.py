# import streamlit as st
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# if torch.cuda.is_available():
#     deviceoption = st.sidebar.radio("Select compute Device.", [
#                                     'cpu', 'cuda'], disabled=False, index=1)
# else:
#     deviceoption = st.sidebar.radio("Select compute Device.", [
#                                     'cpu', 'cuda'], disabled=True, index=0)


# wget.download("https://github.com/NTECAI/AIDrugPOCWith3Layers/tree/main/models/*", out="models/")

# # Function to load a specified model
# def load_model(model_name):
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'models/{model_name}', force_reload=True, device=deviceoption)
#     model.eval()
#     return model

# # Load shape identification model
# shape_model = load_model('shape_best.pt')

# # Load color identification model
# color_model = load_model('color_best.pt')

# # Define the list of models
# models = {
#     ('Green', 'Round'): ['green_round_best.pt'],
#     ('Orange', 'Capsule'): ['orange_capsule_best.pt'],
#     ('Orange', 'Round'): ['orange_round_1_best.pt', 'orange_round_2_best.pt'],
#     ('White', 'Capsule'): ['white_capsule_best.pt'],
#     ('White', 'Round'): ['white_round_1_best.pt', 'white_round_2_best.pt', 'white_round_3_best.pt', 'white_round_4_best.pt', 'white_round_5_best.pt', 'white_round_6_best.pt', 'white_round_7_best.pt'],
# }

# # Function to identify objects based on a given model
# def identify_objects(image, model):
#     # Convert PIL image to OpenCV format
#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Run object identification
#     results = model(image)

#     # Process results
#     objects = []
#     for result in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = result
#         name = model.names[int(cls)]
#         if conf.item() >0.6:
#             objects.append([name, conf.item(), (x1, y1, x2, y2)])

#     return objects

# def add_background(image):
#     # Create a new image with the desired background color
#     new_image = Image.new("RGB", image.size, "black")

#     # Paste the original image onto the new image
#     new_image.paste(image, (0, 0))

#     return new_image

# def draw_bounding_boxes(image, objects):

#     image = add_background(image)

#     # Convert PIL image to NumPy array
#     image_array = np.array(image)

#     # Create a copy of the image for drawing bounding boxes
#     output_image = image_array.copy()

#     # Draw bounding boxes on the output image
#     for obj in objects:
#         name, bbox = obj[0], obj[1]
#         x1, y1, x2, y2 = bbox

#         cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
#         cv2.putText(output_image, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

#     # Convert the output image back to PIL format
#     output_image = Image.fromarray(output_image)

#     return output_image

# # Streamlit app
# def main():
#     st.title("Object Identification")

#     # File uploader
#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Read image
#         image = Image.open(uploaded_file)

#         # Perform shape identification
#         shapes = identify_objects(image, shape_model)

#         # Perform color identification
#         colors = identify_objects(image, color_model)

#         # Display results
#         st.header("1. Shapes:")
#         for shape in shapes[:1]:
#             st.write(shape[:2])

#         st.header("2. Colors:")
#         for color in colors[:1]:
#             st.write(color[:2])

#         # Call the specified model based on color and shape combination
#         color_label = colors[0][0] if colors else None
#         shape_label = shapes[0][0] if shapes else None
        
    
#         if color_label and shape_label and (color_label, shape_label) in models:
#             model_names = models[(color_label, shape_label)]
#             st.header("3. Identified Drugs:")
#             for model_name in model_names:
#                 loaded_model = load_model(model_name)
#                 drugs = identify_objects(image, loaded_model)
#                 # threshold control
#                 if drugs:
#                     st.write(model_name, drugs[:1][0][:2])

#                     # Prepare objects for drawing bounding boxes
#                     objects = [(drug[0], drug[2]) for drug in drugs]
#                     # st.write(objects)
#                     # Draw bounding boxes on the image
#                     output_image = draw_bounding_boxes(image, objects)

#                     # Display the image with bounding boxes
#                     st.image(output_image, channels="RGB")
#         else:
#             st.error("No model found.")

#         # Display output image
#         # st.image(image, channels="RGB")


# if __name__ == "__main__":
#     main()


import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
# import wget



# Configurations
CFG_MODEL_PATH = "models/white_round_1_best.pt"
CFG_ENABLE_URL_DOWNLOAD = False
CFG_ENABLE_VIDEO_PREDICTION = False
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://github.com/NTECAI/AIDrugPOCv5/raw/main/models/eDrug.pt"
# End of Configurations
def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
        
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error(
                'Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`  ', icon="⚠️")

    # -- Sidebar
    st.sidebar.title('⚙️ Options')
    datasrc = st.sidebar.radio("Select input source.", [
                               'From example data.', 'Upload your own data.'])

    if CFG_ENABLE_VIDEO_PREDICTION:
        option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    else:
        option = st.sidebar.radio("Select input type.", ['Image'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('📦 YOLOv5 Streamlit Deployment Example')
    st.sidebar.markdown(
        "https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment")

    if option == "Image":
        loadmodel(deviceoption)


# Downlaod Model from url.
@st.cache_resource
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")
        

@st.cache_resource
def loadmodel(device):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=CFG_MODEL_PATH, force_reload=True, device=device)
    return model


if __name__ == '__main__':
    main()
