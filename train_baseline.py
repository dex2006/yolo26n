from pathlib import Path
import yaml

from ultralytics import YOLO


# =========================================================
# PATH
# =========================================================

DATASET_ROOT = Path(
    r"C:\Users\user\Desktop\專題 001\night_data_v3_split"
)

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\night-agent"
)

YOLO_WEIGHTS = Path(
    r"C:\Users\user\Desktop\yolo\yolo26n.pt"
)

YAML_PATH = PROJECT_ROOT / "baseline_raw.yaml"

RUNS_DIR = PROJECT_ROOT / "runs_baseline"


# =========================================================
# DATASET CHECK
# =========================================================

train_images = DATASET_ROOT / "images" / "train"
val_images = DATASET_ROOT / "images" / "val"
test_images = DATASET_ROOT / "images" / "test"

train_labels = DATASET_ROOT / "labels" / "train"
val_labels = DATASET_ROOT / "labels" / "val"
test_labels = DATASET_ROOT / "labels" / "test"


required_paths = [
    train_images,
    val_images,
    test_images,
    train_labels,
    val_labels,
    test_labels,
    YOLO_WEIGHTS,
]


for p in required_paths:
    if not p.exists():
        raise FileNotFoundError(
            f"找不到：{p}"
        )


# =========================================================
# CREATE DATA YAML
# =========================================================

data_yaml = {
    "path": str(DATASET_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",

    "names": {
        0: "person",
        1: "dog",
        2: "car",
        3: "bicycle",
    },
}


with open(
    YAML_PATH,
    "w",
    encoding="utf-8",
) as f:

    yaml.safe_dump(
        data_yaml,
        f,
        allow_unicode=True,
        sort_keys=False,
    )


print("=" * 70)
print("RAW YOLO26n BASELINE")
print("=" * 70)

print(f"Dataset : {DATASET_ROOT}")
print(f"YAML    : {YAML_PATH}")
print(f"Weight  : {YOLO_WEIGHTS}")

print()

print("正式設定：")
print("epochs = 100")
print("imgsz  = 960")
print("batch  = 8")
print("seed   = 42")
print("device = 0")

print("=" * 70)


# =========================================================
# MODEL
# =========================================================

model = YOLO(
    str(YOLO_WEIGHTS)
)


# =========================================================
# TRAIN
# =========================================================

results = model.train(

    data=str(YAML_PATH),

    epochs=100,

    imgsz=960,

    batch=8,

    device=0,

    workers=0,

    seed=42,

    deterministic=True,

    pretrained=True,

    optimizer="auto",

    patience=30,

    cache=False,

    amp=True,

    plots=True,

    val=True,

    project=str(RUNS_DIR),

    name="raw_yolo26n",

    exist_ok=True,
)


print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    RUNS_DIR
    / "raw_yolo26n"
    / "weights"
    / "best.pt"
)

print("=" * 70)