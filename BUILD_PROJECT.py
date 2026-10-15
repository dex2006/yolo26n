from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\night-agent")

files = {
"config.json": r'''
{
  "yolo_weights": "C:\\Users\\user\\Desktop\\yolo\\yolo26n.pt",
  "mllm_backend": "transformers",
  "transformers_model": "Qwen/Qwen3-VL-4B-Instruct",
  "ollama_model": "qwen3-vl:8b-instruct",
  "output_root": "outputs",
  "staging_root": "staging_dataset",
  "yolo_conf": 0.25,
  "yolo_imgsz": 960
}
''',

"prompt.txt": r'''
你是夜間監視器影像的前處理決策 Agent。

你的目的不是偵測人物，也不是直接修改圖片。
你的工作只有：觀察圖片與影像統計後，決定 OpenCV 應該如何處理，
讓後面的固定 YOLO 更容易偵測小人物，同時避免增加過多誤判。

只能選擇：

none
gamma
clahe
gamma_clahe
sharpen

輸出只能是 JSON：

{
  "action": "gamma_clahe",
  "gamma": 0.80,
  "clahe_clip": 2.0,
  "sharpen": 0.6,
  "reason": "簡短說明原因"
}

限制：

gamma：0.45 ~ 1.20
clahe_clip：1.0 ~ 4.0
sharpen：0.0 ~ 2.0

在本系統中 gamma < 1 代表提亮。

不要因為夜間影像就全部選 Gamma。
不要固定每張都選同一方法。
圖片本身已經清楚時可以選 none。
過度增強可能讓背景紋理造成 YOLO false positive。
''',

"requirements.txt": r'''
ultralytics
opencv-python
numpy
Pillow
gradio
requests
pydantic
transformers
accelerate
bitsandbytes
''',

"INSTALL.bat": r'''@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PY=C:\Users\user\Desktop\yolo\.venv\Scripts\python.exe

echo.
echo ===== Night Agent Install =====
echo Python: %PY%
echo.

"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -U ultralytics gradio opencv-python Pillow requests pydantic
"%PY%" -m pip install -U transformers accelerate bitsandbytes

echo.
echo ===== Check GPU =====
"%PY%" -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo.
echo 安裝完成
pause
''',

"START_AGENT.bat": r'''@echo off
chcp 65001 >nul
cd /d "%~dp0"

"C:\Users\user\Desktop\yolo\.venv\Scripts\python.exe" app.py

pause
''',

"train_yolo26n.py": r'''
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
''',

"app.py": r'''
from pathlib import Path
import json
import re
import io
import base64
import shutil
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
import requests

from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

CONFIG = json.loads(
    (ROOT / "config.json").read_text(
        encoding="utf-8"
    )
)

PROMPT = (
    ROOT / "prompt.txt"
).read_text(
    encoding="utf-8"
)


# ==================================================
# MLLM JSON Schema
# ==================================================

class Decision(BaseModel):

    action: Literal[
        "none",
        "gamma",
        "clahe",
        "gamma_clahe",
        "sharpen"
    ]

    gamma: float = Field(
        default=0.8,
        ge=0.45,
        le=1.20
    )

    clahe_clip: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0
    )

    sharpen: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0
    )

    reason: str = ""


# ==================================================
# Cache
# ==================================================

YOLO_CACHE = {}

HF_MODEL = None
HF_PROCESSOR = None


# ==================================================
# Logging
# ==================================================

def log_event(data):

    folder = (
        ROOT
        / CONFIG["output_root"]
        / "logs"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    path = folder / (
        "agent_"
        + datetime.now().strftime("%Y%m%d")
        + ".jsonl"
    )

    record = {
        "time":
            datetime.now().isoformat(
                timespec="seconds"
            ),
        **data
    }

    with path.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ==================================================
# Image
# ==================================================

def read_image(path):

    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise RuntimeError(
            f"讀不到圖片：{path}"
        )

    return img


def save_image(path, img):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ok, buf = cv2.imencode(
        ".png",
        img
    )

    if not ok:
        raise RuntimeError(
            "圖片存檔失敗"
        )

    buf.tofile(
        str(path)
    )


def get_stats(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return {

        "brightness":
            round(
                float(gray.mean()),
                2
            ),

        "contrast":
            round(
                float(gray.std()),
                2
            ),

        "dark_ratio":
            round(
                float(
                    np.mean(gray < 50)
                ),
                4
            ),

        "sharpness":
            round(
                float(
                    cv2.Laplacian(
                        gray,
                        cv2.CV_64F
                    ).var()
                ),
                2
            )
    }


# ==================================================
# Enhancement
# ==================================================

def gamma_process(
    img,
    gamma
):

    table = np.array([

        (
            (i / 255.0)
            ** gamma
        ) * 255

        for i in range(256)

    ]).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return cv2.LUT(
        img,
        table
    )


def clahe_process(
    img,
    clip
):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    c = cv2.createCLAHE(
        clipLimit=float(clip),
        tileGridSize=(8, 8)
    )

    l = c.apply(l)

    return cv2.cvtColor(
        cv2.merge(
            (l, a, b)
        ),
        cv2.COLOR_LAB2BGR
    )


def sharpen_process(
    img,
    strength
):

    if strength <= 0:
        return img.copy()

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.0
    )

    return cv2.addWeighted(
        img,
        1 + strength,
        blur,
        -strength,
        0
    )


def enhance(
    img,
    action,
    gamma,
    clip,
    sharpen
):

    if action == "none":
        return img.copy()

    if action == "gamma":
        return gamma_process(
            img,
            gamma
        )

    if action == "clahe":
        return clahe_process(
            img,
            clip
        )

    if action == "gamma_clahe":

        img2 = gamma_process(
            img,
            gamma
        )

        return clahe_process(
            img2,
            clip
        )

    if action == "sharpen":

        return sharpen_process(
            img,
            sharpen
        )

    raise ValueError(
        action
    )


# ==================================================
# JSON Parse
# ==================================================

def parse_json(text):

    try:

        obj = json.loads(
            text
        )

    except:

        match = re.search(
            r"\{.*\}",
            text,
            re.S
        )

        if not match:

            raise ValueError(
                "MLLM 沒有輸出 JSON"
            )

        obj = json.loads(
            match.group(0)
        )

    decision = (
        Decision.model_validate(
            obj
        )
    )

    return decision


# ==================================================
# Ollama Backend
# ==================================================

def ollama_decide(
    rgb,
    stats
):

    pil = Image.fromarray(
        rgb.astype(
            np.uint8
        )
    )

    buffer = io.BytesIO()

    pil.save(
        buffer,
        format="JPEG",
        quality=88
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode()

    prompt = (
        PROMPT
        + "\n\n影像統計：\n"
        + json.dumps(
            stats,
            ensure_ascii=False,
            indent=2
        )
    )

    payload = {

        "model":
            CONFIG["ollama_model"],

        "messages": [

            {
                "role":
                    "user",

                "content":
                    prompt,

                "images": [
                    encoded
                ]
            }

        ],

        "stream":
            False,

        "format":
            "json",

        "options": {
            "temperature": 0
        }
    }

    r = requests.post(

        "http://127.0.0.1:11434/api/chat",

        json=payload,

        timeout=300
    )

    r.raise_for_status()

    raw = (
        r.json()
        ["message"]
        ["content"]
    )

    return (
        parse_json(raw),
        raw
    )


# ==================================================
# Transformers Backend
# ==================================================

def load_hf():

    global HF_MODEL
    global HF_PROCESSOR

    if HF_MODEL is not None:

        return (
            HF_MODEL,
            HF_PROCESSOR
        )

    import torch

    from transformers import (
        AutoProcessor,
        AutoModelForMultimodalLM,
        BitsAndBytesConfig
    )

    model_id = (
        CONFIG[
            "transformers_model"
        ]
    )

    quant = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type=
                "nf4",

            bnb_4bit_use_double_quant=
                True,

            bnb_4bit_compute_dtype=
                torch.float16
        )
    )

    HF_PROCESSOR = (
        AutoProcessor.from_pretrained(
            model_id
        )
    )

    HF_MODEL = (
        AutoModelForMultimodalLM
        .from_pretrained(

            model_id,

            device_map="auto",

            quantization_config=
                quant,

            torch_dtype=
                torch.float16
        )
    )

    return (
        HF_MODEL,
        HF_PROCESSOR
    )


def transformers_decide(
    rgb,
    stats
):

    import torch

    model, processor = load_hf()

    pil = Image.fromarray(
        rgb.astype(
            np.uint8
        )
    )

    text = (
        PROMPT
        + "\n\n影像統計：\n"
        + json.dumps(
            stats,
            ensure_ascii=False,
            indent=2
        )
    )

    messages = [

        {
            "role":
                "user",

            "content": [

                {
                    "type":
                        "image",

                    "image":
                        pil
                },

                {
                    "type":
                        "text",

                    "text":
                        text
                }
            ]
        }
    ]

    inputs = (
        processor.apply_chat_template(

            messages,

            add_generation_prompt=True,

            tokenize=True,

            return_dict=True,

            return_tensors="pt"

        ).to(
            model.device
        )
    )

    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=180,

            do_sample=False
        )

    new_tokens = output[
        :,
        inputs[
            "input_ids"
        ].shape[1]:
    ]

    raw = (
        processor.batch_decode(

            new_tokens,

            skip_special_tokens=True

        )[0]
    )

    return (
        parse_json(raw),
        raw
    )


# ==================================================
# UI callbacks
# ==================================================

def load_path(path):

    path = str(path).strip()

    img = read_image(
        path
    )

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    stats = get_stats(
        img
    )

    return (
        rgb,
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2
        ),
        path
    )


def analyze(
    rgb,
    backend
):

    if rgb is None:

        raise gr.Error(
            "請先載入圖片"
        )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    stats = get_stats(
        bgr
    )

    try:

        if backend == "ollama":

            decision, raw = (
                ollama_decide(
                    rgb,
                    stats
                )
            )

        else:

            decision, raw = (
                transformers_decide(
                    rgb,
                    stats
                )
            )

    except Exception as e:

        raise gr.Error(
            str(e)
        )

    d = decision.model_dump()

    log_event({

        "type":
            "mllm",

        "backend":
            backend,

        "stats":
            stats,

        "decision":
            d,

        "raw":
            raw
    })

    return (

        d["action"],
        d["gamma"],
        d["clahe_clip"],
        d["sharpen"],

        json.dumps(
            d,
            ensure_ascii=False,
            indent=2
        ),

        raw
    )


def preview(
    rgb,
    action,
    gamma,
    clip,
    sharpen
):

    if rgb is None:

        raise gr.Error(
            "請先載入圖片"
        )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    out = enhance(

        bgr,
        action,
        gamma,
        clip,
        sharpen
    )

    log_event({

        "type":
            "enhancement",

        "action":
            action,

        "gamma":
            gamma,

        "clahe_clip":
            clip,

        "sharpen":
            sharpen
    })

    return cv2.cvtColor(
        out,
        cv2.COLOR_BGR2RGB
    )


def yolo_predict(
    rgb,
    weights,
    conf,
    imgsz
):

    if rgb is None:

        raise gr.Error(
            "請先產生處理後圖片"
        )

    weights = (
        weights.strip()
    )

    if weights not in YOLO_CACHE:

        YOLO_CACHE[
            weights
        ] = YOLO(
            weights
        )

    model = YOLO_CACHE[
        weights
    ]

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    result = model.predict(

        source=bgr,

        conf=float(conf),

        imgsz=int(imgsz),

        device=0,

        verbose=False
    )[0]

    annotated = result.plot()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls = int(
                box.cls.item()
            )

            detections.append({

                "class_id":
                    cls,

                "class_name":
                    result.names.get(
                        cls,
                        str(cls)
                    ),

                "confidence":
                    round(
                        float(
                            box.conf.item()
                        ),
                        4
                    ),

                "xyxy": [
                    round(
                        float(v),
                        2
                    )
                    for v
                    in box.xyxy[
                        0
                    ].tolist()
                ]
            })

    log_event({

        "type":
            "yolo",

        "weights":
            weights,

        "conf":
            conf,

        "imgsz":
            imgsz,

        "detections":
            detections
    })

    return (

        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        ),

        json.dumps(
            detections,
            ensure_ascii=False,
            indent=2
        )
    )


def find_label(
    source
):

    if not source:

        return None

    p = Path(
        source
    )

    parts = list(
        p.parts
    )

    for i, item in enumerate(
        parts
    ):

        if item.lower() == "images":

            new_parts = list(
                parts
            )

            new_parts[i] = (
                "labels"
            )

            label = Path(
                *new_parts
            ).with_suffix(
                ".txt"
            )

            if label.exists():

                return label

    return None


def save_pair(
    rgb,
    source_path,
    action,
    gamma,
    clip,
    sharpen
):

    if rgb is None:

        raise gr.Error(
            "沒有處理後圖片"
        )

    label = find_label(
        source_path
    )

    if label is None:

        raise gr.Error(
            "找不到這張原圖的 YOLO label，因此不能當訓練資料。"
        )

    folder = (
        ROOT
        / CONFIG["staging_root"]
    )

    image_dir = (
        folder
        / "images"
        / "train"
    )

    label_dir = (
        folder
        / "labels"
        / "train"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    source = Path(
        source_path
    )

    name = (
        source.stem
        + "__"
        + action
        + "__"
        + stamp
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    img_out = (
        image_dir
        / (name + ".png")
    )

    label_out = (
        label_dir
        / (name + ".txt")
    )

    save_image(
        img_out,
        bgr
    )

    shutil.copy2(
        label,
        label_out
    )

    log_event({

        "type":
            "training_pair",

        "source":
            source_path,

        "image":
            str(img_out),

        "label":
            str(label_out),

        "action":
            action,

        "gamma":
            gamma,

        "clahe_clip":
            clip,

        "sharpen":
            sharpen
    })

    return (
        "完成\n"
        + str(img_out)
        + "\n"
        + str(label_out)
    )


# ==================================================
# Gradio Agent UI
# ==================================================

with gr.Blocks(
    title=
        "Night MLLM → OpenCV → YOLO26n Agent"
) as demo:

    gr.Markdown(
        """
# Night Image Enhancement Agent

**MLLM → OpenCV → YOLO26n**

MLLM 只負責決策。  
OpenCV 真正修改圖片。  
所有參數都能手動覆寫。
"""
    )

    source_path = gr.Textbox(

        label=
            "原始資料集圖片完整路徑"
    )

    load_button = gr.Button(
        "載入圖片"
    )

    with gr.Row():

        original = gr.Image(
            label="原圖",
            type="numpy"
        )

        processed = gr.Image(
            label="處理後圖片",
            type="numpy"
        )

        yolo_image = gr.Image(
            label="YOLO 結果",
            type="numpy"
        )

    stats_box = gr.Code(
        label="影像統計",
        language="json"
    )

    backend = gr.Radio(

        choices=[
            "transformers",
            "ollama"
        ],

        value=
            CONFIG[
                "mllm_backend"
            ],

        label="MLLM Backend"
    )

    analyze_button = gr.Button(
        "讓 MLLM 分析",
        variant="primary"
    )

    action = gr.Dropdown(

        choices=[
            "none",
            "gamma",
            "clahe",
            "gamma_clahe",
            "sharpen"
        ],

        value="none",

        label=
            "Action（可以手動修改）"
    )

    with gr.Row():

        gamma = gr.Slider(
            0.45,
            1.20,
            value=0.8,
            step=0.01,
            label="Gamma"
        )

        clahe = gr.Slider(
            1.0,
            4.0,
            value=2.0,
            step=0.1,
            label="CLAHE"
        )

        sharpen = gr.Slider(
            0.0,
            2.0,
            value=0.6,
            step=0.05,
            label="Sharpen"
        )

    decision_box = gr.Code(
        label="MLLM JSON",
        language="json"
    )

    raw_box = gr.Textbox(
        label="MLLM 原始輸出",
        lines=3
    )

    preview_button = gr.Button(
        "套用影像處理 / 預覽"
    )

    weights = gr.Textbox(

        value=
            CONFIG[
                "yolo_weights"
            ],

        label="YOLO 權重"
    )

    with gr.Row():

        conf = gr.Slider(
            0.01,
            0.95,
            value=
                CONFIG[
                    "yolo_conf"
                ],
            step=0.01,
            label="Confidence"
        )

        imgsz = gr.Number(
            value=
                CONFIG[
                    "yolo_imgsz"
                ],
            label="imgsz"
        )

    yolo_button = gr.Button(
        "送進 YOLO26n",
        variant="primary"
    )

    detection_box = gr.Code(
        label="YOLO Detection JSON",
        language="json"
    )

    save_button = gr.Button(
        "保存成 YOLO Training Pair"
    )

    save_status = gr.Textbox(
        label="保存結果"
    )

    load_button.click(

        load_path,

        inputs=[
            source_path
        ],

        outputs=[
            original,
            stats_box,
            source_path
        ]
    )

    analyze_button.click(

        analyze,

        inputs=[
            original,
            backend
        ],

        outputs=[
            action,
            gamma,
            clahe,
            sharpen,
            decision_box,
            raw_box
        ]
    )

    preview_button.click(

        preview,

        inputs=[
            original,
            action,
            gamma,
            clahe,
            sharpen
        ],

        outputs=[
            processed
        ]
    )

    yolo_button.click(

        yolo_predict,

        inputs=[
            processed,
            weights,
            conf,
            imgsz
        ],

        outputs=[
            yolo_image,
            detection_box
        ]
    )

    save_button.click(

        save_pair,

        inputs=[
            processed,
            source_path,
            action,
            gamma,
            clahe,
            sharpen
        ],

        outputs=[
            save_status
        ]
    )


if __name__ == "__main__":

    demo.launch(
        inbrowser=True,
        server_name=
            "127.0.0.1"
    )
''',

"README.md": r'''
# Night MLLM → OpenCV → YOLO26n Agent

流程：

原始圖片
→ MLLM 看圖片
→ JSON 決定 enhancement
→ OpenCV 修圖
→ 可人工調參
→ YOLO26n
→ 看偵測結果
→ 保存實驗紀錄

MLLM 不直接修改圖片。

目前允許：

- none
- gamma
- clahe
- gamma_clahe
- sharpen

所有非幾何影像處理不會改變 bounding box 座標，
因此對應 YOLO label 可以保留。

若未來使用 crop / rotate / perspective 等幾何處理，
必須同步修改 bounding box。
''',

"notes/COMMANDS.md": r'''
# 指令

## 啟動舊 venv

C:\Users\user\Desktop\yolo\.venv\Scripts\Activate.ps1

## 啟動 Agent

START_AGENT.bat

## 訓練 YOLO26n person

python train_yolo26n.py --data "你的data.yaml"

## 目前 MLLM

主要候選：
Qwen/Qwen3-VL-4B-Instruct

Transformers + 4bit

比較組：
Ollama qwen3-vl:8b-instruct
''',

"notes/EXPERIMENT_LOG.md": r'''
# Experiment Log

日期：

實驗 ID：

Dataset split：

圖片數：

Seed：

MLLM：

Runtime：

Prompt：

Action：

Gamma：

CLAHE：

Sharpen：

YOLO：

Weights：

imgsz：

conf：

IoU：

TP：

FP：

FN：

Precision：

Recall：

F1：

Oracle Match：

發現：

下一步：
'''
}

for name, content in files.items():

    path = ROOT / name

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8"
    )

for folder in [
    "outputs",
    "staging_dataset",
    "notes"
]:

    (
        ROOT / folder
    ).mkdir(
        parents=True,
        exist_ok=True
    )

print()
print("===================================")
print(" Night Agent 專案建立完成")
print("===================================")
print()
print(ROOT)
print()
print("下一步執行：")
print("  .\\INSTALL.bat")
print()
