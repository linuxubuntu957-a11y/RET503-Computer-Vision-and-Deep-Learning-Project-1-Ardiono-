from ultralytics import YOLO

PT = "runs/detect/train-2/weights/best.pt"
DATA = "dataset_v1/data.yaml"

# 1) Ekspor ke ONNX
onnx_path = YOLO(PT).export(format="onnx")
print("ONNX:", onnx_path)

# 2) Evaluasi model ONNX pada split TEST
m = YOLO(onnx_path, task="detect")
r = m.val(data=DATA, split="test")
print("mAP50-95:", r.box.map, "| mAP50:", r.box.map50)
