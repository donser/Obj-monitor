# Obj-monitor

https://github.com/ultralytics/ultralytics 下载最新的检测模型：Detection YOLOv8s（exp: models/yolov8s.pt）

## 创建虚拟环境
> conda create -n monitor python=3.8
> conda activate monitor
> pip install -r requirements.txt

## 模型转换为onnx
> python pytorch_to_onnx.py

## 模型转为openvino（float、half、FP32、FP16）
### FP32
> mo --progress --input_shape [1,3,640,640] --input_model models/yolov8s.onnx --output_dir models/yolov8s_FP32 --data_type FP32
### float
> mo --progress --input_shape [1,3,640,640] --input_model models/yolov8s.onnx --output_dir models/yolov8s_float --data_type float
### half
> mo --progress --input_shape [1,3,640,640] --input_model models/yolov8s.onnx --output_dir models/yolov8s_half --data_type half
### FP16
> mo --progress --input_shape [1,3,640,640] --input_model models/yolov8s.onnx --output_dir models/yolov8s_FP16 --data_type FP16

## openvino模型转为int8
> python FP32_to_INT8.py (须注意修改coco128.yaml中的路径)

## 测试openvino模型
> python test_openvino.py

## 展示
CPU ：i5-1035G1
FP32 Inference Time: 108.4091329574585 ms Single Image  /   FPS: 9.224315080468509 FPS Single Image
<img src="./output/000000075910_FP32.jpg" >
INT8 Inference Time: 40.07129669189453 ms Single Image   /   FPS: 24.955518851534347 FPS Single Image
<img src="./output000000075910_INT8.jpg" >




