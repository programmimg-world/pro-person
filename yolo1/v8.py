from ultralytics import YOLO

# Loading model
model = YOLO("yolov8n.pt")  
model.train(data="yolo1\conf.yaml", epochs=100)


