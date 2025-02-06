from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # build a new model from scratch

# Use the model
# results = model.train(data="config.yaml", epochs=100)  # train the model

results = model.train(
    data="config.yaml",
    epochs=50,
    imgsz=640,
    batch=-1,
    # device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    # fraction = 0.1, #subset training
    # cache = True #Stores dataset images in RAM | disk, False
    # patience=5 #early stopping: if no improvement after 5 consecutive epochs
    # warmup_epochs=3,  # Number of warmup epochs
    # lr0=0.005,  # Learning rate
    # lrf=0.1,  # Final LR fraction
    # momentum=0.937,  # Controls how much past gradients affect the current gradient update, gama
    # weight_decay=0.0005,  # discourages large weight values to prevent overfitting.
    # optimizer="AdamW",  # Options: 'SGD', 'Adam', 'AdamW', 'RMSprop'
)

# # Evaluate model performance on the validation set
# metrics = model.val()
# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()
# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model