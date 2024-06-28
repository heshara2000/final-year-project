import json
import os
import time
import PIL
from PIL import Image
import ultralytics
import ultralyticsplus
#import streamlit as st
import torch
from ultralyticsplus import YOLO, render_result

from convert import convert_to_braille_unicode, parse_xywh_and_class


# image_path = "../word_detection/alpha-numeric.jpeg"
image_path = "alpha-numeric.jpeg"     # in src
#image_path = "hello.png" 
#image_path = "flask.png" 
#image_path = "a.jpg" 

model_path = "../weights/yolov8_braille.pt"
#alphabet_map_path = "./utils/alphabet_map.json"

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    """load model from path"""
    model = YOLO(model_path)
    return model

def load_image(image_path):
    """Load image from path."""
    try:
        print(f"Attempting to load image from path: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        
        imagefounded = Image.open(image_path)
        # print(f"Image loaded successfully from path: {image_path}")
        return imagefounded
    except Exception as ex:
        print(f"Error loading image: {ex}")
        return None
    

# model_path = "snoop2head/yolov8m-braille"

try:
    #model = load_model(model_path)
    model =YOLO("../weights/yolov8_braille.pt")
    model.overrides["conf"] = 10  # NMS confidence threshold
    model.overrides["iou"] = 0.7  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 1000  # maximum number of detections per image


    # print("Model class names and their indices:")
    # for idx, name in enumerate(model.names):
    #  print(f"{idx}: {name}")

except Exception as ex:
    print("-----------------------------------------------------")
    print(ex)
    print(f"Unable to load model. Check the specified path: {model_path}")

#source_img = None

#  source_img = st.sidebar.file_uploader(
#     "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
# ) 
#image = load_image(source_img)
try:
    image = load_image(image_path)
    if image is None:
        raise ValueError("Image loading failed.")
    
    with torch.no_grad():
        res = model.predict(image, save=True, save_txt=True, exist_ok=True)
        boxes = res[0].boxes  # First image
        res_plotted = res[0].plot()[:, :, ::-1]

        list_boxes = parse_xywh_and_class(boxes)
        
        try:
            if list_boxes:
                print("Not empty")
            else:
                raise ValueError("No boxes detected.")
                
            for box_line in list_boxes:
                str_left_to_right = ""
                box_classes = box_line[:, -1]
                print(f"Detected classes: {box_classes}")

                for each_class in box_classes:
                    str_left_to_right += convert_to_braille_unicode(model.names[int(each_class)])
                
                print(str_left_to_right + "\n")
        except Exception as ex:
            print("Error processing detected boxes.")
            print(ex)
except Exception as ex:
    print("Error during the detection process.")
    print(ex)
#display detect braille patterns
# try:
#     #st.success(f"Done! Inference time: {time.time() - start_time:.2f} seconds")
#     #st.subheader("Detected Braille Patterns")
#     for box_line in list_boxes:
#         str_left_to_right = ""
#         box_classes = box_line[:, -1]
#         #print("=========================================="+ box_classes)


#         for each_class in box_classes:
#             #print("=========================================="+ each_class)
#             str_left_to_right += convert_to_braille_unicode(
#                 model.names[int(each_class)]
#             )
#             print(str_left_to_right + "\n")
#             print("test 1")
            #st.write(str_left_to_right) 
        #st.write(str_left_to_right)
    # st.write("=========================================================================================")
    # st.write(str_left_to_right)
#     # st.write("=========================================================================================")
# except Exception as ex:
#     print("Please try again with images with types of JPG, JPEG, PNG ...")
#     print("check again..")

# IMAGE_DOWNLOAD_PATH = f"runs/detect/predict/image0.jpg"
# with open(IMAGE_DOWNLOAD_PATH, "rb") as fl:
#     st.download_button(
#         "Download object-detected image",
#         data=fl,
#         file_name="image0.jpg",
#         mime="image/jpg",
#     )
