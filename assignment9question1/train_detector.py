from ultralytics import YOLO

if __name__ == "__main__":   # required on Windows when workers > 0
    model = YOLO("yolo11s.pt")
    model.train(data="datasets/yolo/data.yaml", imgsz=640, epochs=100,
                batch=16, device=0, workers=2)
    model.val(split="test", device=0)