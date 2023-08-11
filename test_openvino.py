# Edit by donser
#

import os
import cv2
import time
import torch
import numpy as np
from ultralytics.yolo.utils import ops
from openvino.runtime import Core

def read_txt(txt):
    label = []
    with open(txt) as f:
        line = f.readline()
        while line:
            doc = line.replace("\n","").split(":")
            label.append(doc[1])
            line = f.readline()
    return label

def letterbox(img,new_shape=(640, 640),color=(114, 114, 114),stride=32):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    dw /= 2  
    dh /= 2
    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def preprocessImage(img):
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img
    
def image2tensor(image):
    input_tensor = image.astype(np.float32)
    input_tensor /= 255.0
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

def postprocess(pred_boxes, input_hw, orig_img, min_conf_threshold=0.5, nms_iou_threshold=0.7, agnosting_nms=False, max_detections=30, pred_masks=None, retina_mask=False):
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    if pred_masks is not None:
        nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None
    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results

def inferenceOpenvino(det_model, url, label_txt):
    # Initialize
    core = Core()
    model = core.read_model(det_model)
    config_cpu = {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": "80"}
    config_gpu = {"PERFORMANCE_HINT": "THROUGHPUT"}
    openvinoModel = core.compile_model(model,'CPU',config_cpu)
    #openvinoModel = core.compile_model(model,'GPU',config_gpu)
    
    label = read_txt(label_txt)
    testImage = cv2.imread(url)
    start = time.time()
    preprocessedImage = letterbox(testImage, new_shape=(640,640))[0]
    preprocessedImage = preprocessImage(preprocessedImage)
    inputTensor = image2tensor(preprocessedImage)
    
    #20 loop-times 
    for i in range(100):
        result = openvinoModel(inputTensor)
    
    boxes = result[openvinoModel.output(0)]
    input_hw = inputTensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=testImage)
    
    end = time.time()
    inf_time = (end - start)/100
    print('Inference Time: {} ms Single Image'.format(inf_time*1000))
    fps = 1. / (inf_time)
    print('Estimated Inference FPS: {} FPS Single Image'.format(fps))
    
    for i in detections[0]["det"]:
        cv2.rectangle(testImage, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 1)
        testImage = cv2.putText(testImage, "FPS:"+'{:.2f}'.format(fps), (0+5, testImage.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA, False)
        testImage = cv2.putText(testImage, label[int(i[5])], (int(i[0]), int(i[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA, False)
    
    sit = max(url.rfind("/"),url.rfind("\\"))
    cv2.imwrite("./output/"+url[sit:],testImage)

if __name__=='__main__':
    inferenceOpenvino('models/yolov8s_FP32/yolov8s.xml', "val/000000075910.jpg", "datasets/coco_class.txt")
    inferenceOpenvino('models/yolov8s_Int8/yolov8s.xml', "val/000000075910.jpg", "datasets/coco_class.txt")


