from ultralytics import YOLO


def train_yolo():
    model = YOLO("yolo12s-seg.yaml").load("yolo12s.pt")

    results = model.train(
        data=r"D:\cow.v1i.yolov12\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
    
    )

    print("🎉 การเทรนเสร็จสมบูรณ์!")


if __name__ == "__main__":
    train_yolo()