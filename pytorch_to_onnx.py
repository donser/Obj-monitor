# Edit by donser
#

from ultralytics import YOLO

# Load model
model = YOLO('models/yolov8s.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')
# model.export(format='openvino')