from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # You can switch to yolov8s.pt if GPU allows

    model.train(
        data=r"D:\garbge project\segmentacion de residuos copy.v1i.yolov8-obb\data.yaml",
        epochs=50  ,   
        imgsz=640,
        batch=4,           # ⬅️ Reduce batch size
        workers=0,         # ⬅️ Avoid multiprocessing (helps on Windows)
        device='cuda' if model.device.type == 'cuda' else 'cpu',
        name="garbage_training_lowmem",
        pretrained=True,
        cache=False,       # ⬅️ Avoid caching images in RAM
        mosaic=0.5,        # ⬅️ Use less mosaic augmentation (reduces RAM)
        patience=20,
    )
