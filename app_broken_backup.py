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
        raise RuntimeError("Cannot read image: " + str(path))
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
            "??摮?憭望?"
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
                "MLLM 瘝?頛詨 JSON"
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
        + "\n\n敶勗?蝯梯?嚗n"
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
        Qwen3VLForConditionalGeneration,
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
        Qwen3VLForConditionalGeneration
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
        + "\n\n敶勗?蝯梯?嚗n"
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
            "隢?頛??"
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
            "隢?頛??"
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
            "隢??Ｙ???敺???
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
            "瘝???敺???
        )

    label = find_label(
        source_path
    )

    if label is None:

        raise gr.Error(
            "?曆??圈撐????YOLO label嚗?甇支??賜閮毀鞈???
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
        "摰?\n"
        + str(img_out)
        + "\n"
        + str(label_out)
    )


# ==================================================
# Gradio Agent UI
# ==================================================

with gr.Blocks(
    title=
        "Night MLLM ??OpenCV ??YOLO26n Agent"
) as demo:

    gr.Markdown(
        """
# Night Image Enhancement Agent

**MLLM ??OpenCV ??YOLO26n**

MLLM ?芾?鞎祆捱蝑? 
OpenCV ?迤靽格???? 
????賊?賣???撖怒?
"""
    )

    source_path = gr.Textbox(

        label=
            "??鞈??????渲楝敺?
    )

    load_button = gr.Button(
        "頛??"
    )

    with gr.Row():

        original = gr.Image(
            label="??",
            type="numpy"
        )

        processed = gr.Image(
            label="??敺???,
            type="numpy"
        )

        yolo_image = gr.Image(
            label="YOLO 蝯?",
            type="numpy"
        )

    stats_box = gr.Code(
        label="敶勗?蝯梯?",
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
        "霈?MLLM ??",
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
            "Action嚗隞交??耨?對?"
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
        label="MLLM ??頛詨",
        lines=3
    )

    preview_button = gr.Button(
        "憟敶勗??? / ?汗"
    )

    weights = gr.Textbox(

        value=
            CONFIG[
                "yolo_weights"
            ],

        label="YOLO 甈?"
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
        "??YOLO26n",
        variant="primary"
    )

    detection_box = gr.Code(
        label="YOLO Detection JSON",
        language="json"
    )

    save_button = gr.Button(
        "靽???YOLO Training Pair"
    )

    save_status = gr.Textbox(
        label="靽?蝯?"
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


