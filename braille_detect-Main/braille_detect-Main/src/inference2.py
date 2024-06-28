import PIL.Image
from ultralytics import YOLO
from convert import convert_to_braille_unicode, parse_xywh_and_class

def load_model(model_path):
    """Load model from path"""
    model = YOLO(model_path)
    return model

def load_image(image_path):
    """Load image from path"""
    image = PIL.Image.open(image_path)
    return image

# Constants
CONF = 0.15 # or other desirable confidence threshold level
MODEL_PATH = "./weights/yolov8_braille.pt"
IMAGE_PATH = "./assets/alpha-numeric.jpeg"

# Load image and model
image = load_image(IMAGE_PATH)
model = load_model(MODEL_PATH)

# Receiving results from the model
res = model.predict(image, save=True, save_txt=True, exist_ok=True, conf=CONF)
boxes = res[0].boxes  # Assuming we process the first image

# Parse the boxes
list_boxes = parse_xywh_and_class(boxes)

# Convert to braille unicode and construct the result string
result = ""
for box_line in list_boxes:
    str_left_to_right = ""
    box_classes = box_line[:, -1]  # Assuming box_line is a numpy array
    for each_class in box_classes:
        str_left_to_right += convert_to_braille_unicode(model.names[int(each_class)])
    result += str_left_to_right + "\n"

print(result)
