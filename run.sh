yolo segment train data=gas_seg_store2.yaml model=yolov8m-seg.pt epochs=100 imgsz=640 device=0 batch=80   bgr=0.5  name=gas_seg_20260825_test


yolo segment train data=gas_seg_store2.yaml model=yolo11x-seg.pt epochs=100 imgsz=640 device=0 batch=40   bgr=0.5  name=gas_seg_20260825_test



yolo segment train data=gas_seg_store2.yaml model=yolo26x-seg.pt epochs=50 imgsz=640 device=1 batch=40   bgr=0.5  name=gas_seg_20260825_26

yolo export task=segment model=/store2/cx/gas-seg/runs/segment/gas_seg_20260825_26/weights/best.pt format=onnx 
yolo export model=/store2/cx/gas-seg/runs/segment/gas_seg_20260825_26/weights/best.pt format=onnx opset=12 simplify=True imgsz=640