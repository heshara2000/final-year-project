import json
import numpy as np
import torch


# def convert_to_braille_unicode(str_input: str, path: str = "./utils/braille_map.json") -> str:
#     with open(path, "r") as fl:
#         data = json.load(fl)

#     if str_input in data.keys():
#         str_output = data[str_input]
#         print("convert_to_braille_unicode :::" + str_output)
#     return str_input


def convert_to_braille_unicode(str_input: str, path: str = "./utils/braille_map.json") -> str:
    try:
        with open(path, "r", encoding="utf-8") as fl:
            data = json.load(fl)

        if str_input in data:
            str_output = data[str_input]
            print(f"convert_to_braille_unicode ::: {str_output}")
            # print("convert_to_braille_unicode :::" + str_output)
            return str_output
        else:
            raise KeyError(f"Input '{str_input}' not found in the braille map.")
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {path} is not a valid JSON file.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    return str_input



def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    """
    boxes input tensor
        boxes (torch.Tensor) or (numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6).
        orig_shape (torch.Tensor) or (numpy.ndarray): Original image size, in the format (height, width).
    Properties:
        xyxy (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor) or (numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor) or (numpy.ndarray): The class values of the boxes.
        xywh (torch.Tensor) or (numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor) or (numpy.ndarray): The boxes in xywh format normalized by original image size.
    """

    # copy values from troublesome "boxes" object to numpy array
    new_boxes = np.zeros(boxes.shape)
    new_boxes[:, :4] = boxes.xywh.numpy()  # first 4 channels are xywh
    new_boxes[:, 4] = boxes.conf.numpy()   # 5th channel is confidence
    new_boxes[:, 5] = boxes.cls.numpy()  # 6th channel is class which is last channel

    # sort according to y coordinate
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]

    # find threshold index to break the line
    y_threshold = np.mean(new_boxes[:, 3]) // 2
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]

    # cluster according to threshold_index
    boxes_clustered = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    for cluster in boxes_clustered:
        # sort according to x coordinate
        cluster = cluster[cluster[:, 0].argsort()]
        boxes_return.append(cluster)

    return boxes_return



# def parse_xywh_and_class(boxes: torch.Tensor) -> list:
#     """
#     Parse the bounding boxes and their classes into lines and sort them within each line.
#     """
#     try:
#         print(f"Original boxes shape: {boxes.shape}")
        
#         # Ensure the boxes tensor has the correct shape
#         if boxes.shape[1] < 6:
#             raise ValueError("Expected at least 6 columns in the boxes tensor.")

#         # Copy values from troublesome "boxes" object to numpy array
#         new_boxes = np.zeros(boxes.shape)
#         new_boxes[:, :4] = boxes.xywh.numpy()  # first 4 channels are xywh
#         new_boxes[:, 4] = boxes.conf.numpy()   # 5th channel is confidence
#         new_boxes[:, 5] = boxes.cls.numpy()  # 6th channel is class which is last channel

#         print(f"New boxes (xywh, conf, cls): {new_boxes}")

#         # Sort according to y coordinate
#         new_boxes = new_boxes[new_boxes[:, 1].argsort()]
#         print(f"Boxes sorted by y-coordinate: {new_boxes}")

#         # Find threshold index to break the line
#         y_threshold = np.mean(new_boxes[:, 3]) / 2  # Use division instead of integer division
#         boxes_diff = np.diff(new_boxes[:, 1])
#         threshold_index = np.where(boxes_diff > y_threshold)[0]

#         print(f"Y threshold: {y_threshold}, Threshold indices: {threshold_index}")

#         # Cluster according to threshold_index
#         boxes_clustered = np.split(new_boxes, threshold_index + 1)
#         boxes_return = []
#         for i, cluster in enumerate(boxes_clustered):
#             # Sort according to x coordinate
#             cluster = cluster[cluster[:, 0].argsort()]
#             print(f"Cluster {i} sorted by x-coordinate: {cluster}")
#             boxes_return.append(cluster)

#         return boxes_return

#     except Exception as ex:
#         print(f"An error occurred in parse_xywh_and_class: {ex}")
#         return []

# # Example usage (you can add this part for testing purposes)
# if __name__ == "__main__":
#     # Simulate some boxes for testing
#     # Assuming each box is in the format: [x, y, w, h, confidence, class]
#     dummy_boxes = torch.tensor([
#         [50, 10, 20, 30, 0.9, 0],
#         [30, 50, 20, 30, 0.85, 1],
#         [70, 10, 20, 30, 0.95, 2],
#         [10, 100, 20, 30, 0.8, 3],
#         [90, 100, 20, 30, 0.88, 4]
#     ])

#     result = parse_xywh_and_class(dummy_boxes)
#     print(f"Result: {result}")
