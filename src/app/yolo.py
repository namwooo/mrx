from ultralytics import YOLO

model = YOLO("../../models/yolo11/yolo11n.pt")
result = model.predict("/Users/youngchankim/Projects/mrx/data/000000581918.jpg")
print(result)