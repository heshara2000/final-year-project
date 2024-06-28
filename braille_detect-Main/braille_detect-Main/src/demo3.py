import json
import os
import time
import PIL
from PIL import Image
import ultralytics
import ultralyticsplus
import torch
from ultralyticsplus import YOLO, render_result

from convert import convert_to_braille_unicode, parse_xywh_and_class

# image_path = "../word_detection/alpha-numeric.jpeg"
image_path = "alpha-numeric.jpeg"     # in src
#image_path = "hello.png" 
# image_path = "flask.png" 
# image_path = "a.jpg" 

model_path = "../weights/yolov8_braille.pt"
#alphabet_map_path = "./utils/braille_dict.json"
alphabet_map_path = "./utils/alphabet_map.json"

def load_model(model_path):
    """Load model from path."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

def load_image(image_path):
    """Load image from path."""
    try:
        print(f"Attempting to load image from path: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        
        image = Image.open(image_path)
        return image
    except Exception as ex:
        print(f"Error loading image: {ex}")
        return None

def load_alphabet_map(alphabet_map_path):
    """Load alphabet map from JSON file."""
    try:
        with open(alphabet_map_path, 'r') as file:
            alphabet_map = json.load(file)
        return alphabet_map
    except Exception as ex:
        print(f"Error loading alphabet map: {ex}")
        return None

def create_braille_to_english_mapping(alphabet_map):
    """Create Braille Unicode to English mapping."""
    braille_to_english = {}
    for letter, binary in alphabet_map.items():
        braille_unicode = convert_to_braille_unicode(binary)
        braille_to_english[braille_unicode] = letter
    return braille_to_english

def convert_braille_to_english(braille_string, braille_to_english):
    """Convert Braille string to English using the provided mapping."""
    english_string = ""
    for char in braille_string:
        english_string += braille_to_english.get(char, "?")  # Use '?' for unknown characters
    return english_string

# Load the model
try:
    model = load_model(model_path)
    model.overrides["conf"] = 1  # NMS confidence threshold
    model.overrides["iou"] = 0.2  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 1000  # maximum number of detections per image

except Exception as ex:
    print("-----------------------------------------------------")
    print(ex)
    print(f"Unable to load model. Check the specified path: {model_path}")

# Load the image
try:
    image = load_image(image_path)
    if image is None:
        raise ValueError("Image loading failed.")
    
    # Load the alphabet map and create the Braille to English mapping
    alphabet_map = load_alphabet_map(alphabet_map_path)
    if alphabet_map is None:
        raise ValueError("Alphabet map loading failed.")
    
    braille_to_english = create_braille_to_english_mapping(alphabet_map)

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
                    braille_char = convert_to_braille_unicode(model.names[int(each_class)])
                    print(f"convert_to_braille_unicode ::: {braille_char}")
                    str_left_to_right += braille_char
                
                print(str_left_to_right + "\n")

                # Convert Braille to English
                english_translation = convert_braille_to_english(str_left_to_right, braille_to_english)
                print(f"Converted to English: {english_translation}")
        except Exception as ex:
            print("Error processing detected boxes.")
            print(ex)
except Exception as ex:
    print("Error during the detection process.")
    print(ex)
