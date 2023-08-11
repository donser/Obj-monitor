# Edit by donser
#

import os
import yaml
import nncf
import openvino
import numpy as np
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader

def create_data_source():
    data = check_det_dataset('datasets/coco128.yaml')
    val_dataloader = create_dataloader(data['val'], imgsz=640, batch_size=1, stride=32, pad=0.5, workers=1)[0]
    return val_dataloader

def transform_fn(data_item):
    images = data_item['img']
    images = images.float()
    images = images / 255.0
    images = images.cpu().detach().numpy()
    return images
        
if __name__ == "__main__":
    # models
    FP32_path = "models/yolov8s_FP32/yolov8s.xml"
    Int8_path = "models/yolov8s_Int8/yolov8s.xml"

    # create data
    data_source = create_data_source()

    #check data
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)

    # QuantizationPreset
    subset_size = 40
    preset = nncf.QuantizationPreset.MIXED

    # deal
    core = openvino.runtime.Core()
    model = core.read_model(FP32_path)
    quantized_model = nncf.quantize(model, 
                                    nncf_calibration_dataset, 
                                    preset=preset, 
                                    subset_size=subset_size
                                    )
    openvino.runtime.serialize(quantized_model, Int8_path)
    print("Succeed to Openvino Int8")

