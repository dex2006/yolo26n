import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    required=True,
    help="YOLO data.yaml 完整路徑"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100
)

parser.add_argument(
    "--imgsz",
    type=int,
    default=960
)

args = parser.parse_args()

model = YOLO(
    r"C:\Users\user\Desktop\yolo\yolo26n.pt"
)

print("使用 YOLO26n")
print("Dataset:", args.data)

model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=-1,
    device=0,
    workers=4,
    classes=[0],
    patience=25,
    amp=True,
    seed=42,
    project="runs_yolo26",
    name="person_yolo26n"
)
