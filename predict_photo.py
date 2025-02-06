import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pre-trained model
# model = YOLO("yolo11x.pt") 
model = YOLO("./runs/detect/train2/weights/best.pt")

# Load the image
image_path = "me_and_apple.jpg"

# Perform object detection
results = model(image_path)

# Show image with bounding boxes
for result in results:
    result.show()  

# Save output
output_path = "output.jpg"
results[0].save(output_path)


output_image = cv2.imread(output_path)
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.imshow(output_image)
plt.axis("off")
plt.show()
