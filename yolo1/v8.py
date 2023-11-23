from ultralytics import YOLO

# Loading model
model = YOLO("yolov8n.yaml")  
model.train(data="yolo1\conf.yaml", epochs=90)


