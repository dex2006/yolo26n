from pathlib import Path
import json
import hashlib

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

YOLO_WEIGHTS = Path(
    r"C:\Users\user\Desktop\yolo\yolo26n.pt"
)

YOLO_CONF = 0.25
YOLO_IMGSZ = 960


PRESETS = {

    "N": {
        "name": "不處理",
        "action": "none",
        "gamma": None,
        "clahe": None,
        "sharpen": None,
    },

    "G1": {
        "name": "強 Gamma",
        "action": "gamma",
        "gamma": 0.65,
        "clahe": None,
        "sharpen": None,
    },

    "G2": {
        "name": "輕 Gamma",
        "action": "gamma",
        "gamma": 0.80,
        "clahe": None,
        "sharpen": None,
    },

    "C1": {
        "name": "輕 CLAHE",
        "action": "clahe",
        "gamma": None,
        "clahe": 1.5,
        "sharpen": None,
    },

    "C2": {
        "name": "強 CLAHE",
        "action": "clahe",
        "gamma": None,
        "clahe": 2.5,
        "sharpen": None,
    },

    "GC1": {
        "name": "強 Gamma + CLAHE",
        "action": "gamma_clahe",
        "gamma": 0.65,
        "clahe": 2.0,
        "sharpen": None,
    },

    "GC2": {
        "name": "輕 Gamma + CLAHE",
        "action": "gamma_clahe",
        "gamma": 0.80,
        "clahe": 2.0,
        "sharpen": None,
    },

    "S1": {
        "name": "輕 Sharpen",
        "action": "sharpen",
        "gamma": None,
        "clahe": None,
        "sharpen": 0.50,
    },

    "S2": {
        "name": "強 Sharpen",
        "action": "sharpen",
        "gamma": None,
        "clahe": None,
        "sharpen": 1.00,
    },
}


# =========================================================
# GLOBAL MODELS
# =========================================================

mllm_model = None
mllm_processor = None
yolo_model = None


# =========================================================
# IMAGE
# =========================================================

def ensure_rgb(img):

    if img is None:
        raise ValueError(
            "請先上傳圖片"
        )

    arr = np.asarray(img)

    if arr.ndim == 2:

        arr = cv2.cvtColor(
            arr,
            cv2.COLOR_GRAY2RGB,
        )

    if arr.shape[2] == 4:

        arr = cv2.cvtColor(
            arr,
            cv2.COLOR_RGBA2RGB,
        )

    return arr.astype(
        np.uint8
    )


def image_hash(img):

    rgb = ensure_rgb(
        img
    )

    return hashlib.sha256(
        rgb.tobytes()
    ).hexdigest()[:16]


def calculate_stats(img):

    rgb = ensure_rgb(
        img
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    contrast_std = float(
        np.std(gray)
    )

    dark_ratio = float(
        np.mean(
            gray < 50
        )
    )

    overexposure_ratio = float(
        np.mean(
            gray > 230
        )
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    median = cv2.medianBlur(
        gray,
        3,
    )

    noise_estimate = float(
        np.mean(
            np.abs(
                gray.astype(
                    np.float32
                )
                -
                median.astype(
                    np.float32
                )
            )
        )
    )

    return {

        "brightness":
            round(
                brightness,
                2,
            ),

        "contrast_std":
            round(
                contrast_std,
                2,
            ),

        "dark_ratio":
            round(
                dark_ratio,
                4,
            ),

        "overexposure_ratio":
            round(
                overexposure_ratio,
                4,
            ),

        "sharpness":
            round(
                sharpness,
                2,
            ),

        "noise_estimate":
            round(
                noise_estimate,
                2,
            ),

        "image_sha256_short":
            image_hash(
                rgb
            ),
    }


# =========================================================
# MLLM LOAD
# =========================================================

def load_mllm():

    global mllm_model
    global mllm_processor

    if mllm_model is not None:
        return

    print()
    print("=" * 60)
    print("Loading Qwen3-VL-8B")
    print("4-bit NF4")
    print("=" * 60)

    quant = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_use_double_quant=True,

        bnb_4bit_compute_dtype=torch.float16,
    )

    mllm_processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    mllm_model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(

            MODEL_ID,

            device_map="auto",

            quantization_config=quant,

            dtype=torch.float16,

            low_cpu_mem_usage=True,
        )
    )

    mllm_model.eval()

    print(
        "MLLM READY"
    )


# =========================================================
# JSON PARSER
# =========================================================

def parse_json(raw):

    start = raw.find(
        "{"
    )

    end = raw.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
    ):

        raise ValueError(
            "MLLM 沒有輸出 JSON"
        )

    obj = json.loads(
        raw[
            start:
            end + 1
        ]
    )

    required = [

        "darkness",

        "low_contrast",

        "blur",

        "noise",

        "overexposure",

        "small_object_visibility",

        "reason",
    ]

    for key in required:

        if key not in obj:

            raise ValueError(
                f"缺少欄位：{key}"
            )

    score_keys = [

        "darkness",

        "low_contrast",

        "blur",

        "noise",

        "overexposure",

        "small_object_visibility",
    ]

    for key in score_keys:

        obj[key] = int(
            float(
                obj[key]
            )
        )

        obj[key] = max(
            0,
            min(
                100,
                obj[key],
            )
        )

    obj["reason"] = str(
        obj["reason"]
    )

    return obj


# =========================================================
# MLLM ANALYSIS
# =========================================================

def ask_mllm(img):

    load_mllm()

    rgb = ensure_rgb(
        img
    )

    stats = calculate_stats(
        rgb
    )

    pil = Image.fromarray(
        rgb
    )

    prompt = f"""
You are an image-quality diagnostic module for a computer-vision
research project involving nighttime small-object detection.

IMPORTANT:

You are NOT allowed to choose an image enhancement method.

Do NOT choose:
Gamma
CLAHE
Sharpen
GC1
GC2
G1
G2
C1
C2
S1
S2

Your only job is to diagnose THIS image.

The downstream task is YOLO object detection.

Judge image quality based on whether useful object details
and small-object edges remain visible.

Computer-measured statistics:

{json.dumps(stats, indent=2)}

Give each score from 0 to 100.

darkness:
0 = very bright
100 = extremely dark

low_contrast:
0 = strong useful contrast
100 = extremely poor contrast

blur:
0 = sharp
100 = extremely blurry

noise:
0 = clean
100 = extremely noisy

overexposure:
0 = no overexposure
100 = severe overexposure

small_object_visibility:
0 = tiny objects almost impossible to identify
100 = tiny objects clearly visible

reason:
Return ONE short Traditional Chinese sentence
describing the main visual-quality issue.

Return JSON ONLY.

Exact format:

{{
  "darkness": 0,
  "low_contrast": 0,
  "blur": 0,
  "noise": 0,
  "overexposure": 0,
  "small_object_visibility": 0,
  "reason": "..."
}}
"""

    messages = [

        {
            "role": "user",

            "content": [

                {
                    "type": "image",
                    "image": pil,
                },

                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = (
        mllm_processor
        .apply_chat_template(

            messages,

            tokenize=True,

            add_generation_prompt=True,

            return_dict=True,

            return_tensors="pt",
        )
    )

    inputs = inputs.to(
        mllm_model.device
    )

    with torch.inference_mode():

        output = (
            mllm_model.generate(

                **inputs,

                max_new_tokens=180,

                do_sample=False,

                use_cache=True,
            )
        )

    input_len = (
        inputs[
            "input_ids"
        ]
        .shape[1]
    )

    generated = output[
        :,
        input_len:
    ]

    raw = (
        mllm_processor
        .batch_decode(

            generated,

            skip_special_tokens=True,

            clean_up_tokenization_spaces=False,

        )[0]
        .strip()
    )

    diagnosis = parse_json(
        raw
    )

    return (
        diagnosis,
        raw,
        stats,
    )


# =========================================================
# POLICY
# =========================================================

def choose_preset(d):

    darkness = d[
        "darkness"
    ]

    contrast = d[
        "low_contrast"
    ]

    blur = d[
        "blur"
    ]

    noise = d[
        "noise"
    ]

    over = d[
        "overexposure"
    ]

    # -------------------------
    # Overexposed
    # -------------------------

    if over >= 70:

        if contrast >= 65:

            return (
                "C1",
                "畫面有明顯過曝，"
                "避免繼續提亮，只做輕度 CLAHE。",
            )

        return (
            "N",
            "畫面已有過曝風險，"
            "保留原圖。",
        )

    # -------------------------
    # Very dark + low contrast
    # -------------------------

    if (
        darkness >= 75
        and contrast >= 65
    ):

        return (
            "GC1",
            "畫面非常暗且低對比，"
            "使用較強 Gamma + CLAHE。",
        )

    # -------------------------
    # Moderate dark + contrast
    # -------------------------

    if (
        darkness >= 55
        and contrast >= 55
    ):

        return (
            "GC2",
            "畫面偏暗且低對比，"
            "使用較溫和 Gamma + CLAHE。",
        )

    # -------------------------
    # Darkness mainly
    # -------------------------

    if darkness >= 75:

        return (
            "G1",
            "主要問題為亮度不足，"
            "使用較強 Gamma。",
        )

    if darkness >= 55:

        return (
            "G2",
            "畫面稍暗，"
            "使用溫和 Gamma。",
        )

    # -------------------------
    # Contrast mainly
    # -------------------------

    if contrast >= 75:

        return (
            "C2",
            "亮度尚可但低對比明顯，"
            "使用較強 CLAHE。",
        )

    if contrast >= 55:

        return (
            "C1",
            "亮度尚可但對比稍弱，"
            "使用輕度 CLAHE。",
        )

    # -------------------------
    # Blur
    # -------------------------

    if (
        blur >= 75
        and noise < 55
    ):

        return (
            "S2",
            "主要問題為明顯模糊，"
            "且雜訊不高，使用較強銳化。",
        )

    if (
        blur >= 55
        and noise < 55
    ):

        return (
            "S1",
            "畫面稍微模糊，"
            "使用輕度銳化。",
        )

    # -------------------------
    # No processing
    # -------------------------

    return (
        "N",
        "未偵測到需要明顯增強的問題，"
        "保留原圖。",
    )


# =========================================================
# OPENCV
# =========================================================

def apply_gamma(
    bgr,
    gamma,
):

    table = np.array(

        [
            (
                (i / 255.0)
                ** gamma
            )
            * 255

            for i in range(
                256
            )
        ]

    ).astype(
        np.uint8
    )

    return cv2.LUT(
        bgr,
        table,
    )


def apply_clahe(
    bgr,
    clip,
):

    lab = cv2.cvtColor(

        bgr,

        cv2.COLOR_BGR2LAB,
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(

        clipLimit=float(
            clip
        ),

        tileGridSize=(
            8,
            8,
        ),
    )

    l2 = clahe.apply(
        l
    )

    merged = cv2.merge(
        [
            l2,
            a,
            b,
        ]
    )

    return cv2.cvtColor(

        merged,

        cv2.COLOR_LAB2BGR,
    )


def apply_sharpen(
    bgr,
    amount,
):

    blurred = cv2.GaussianBlur(

        bgr,

        (
            0,
            0,
        ),

        1.0,
    )

    return cv2.addWeighted(

        bgr,

        1.0 + amount,

        blurred,

        -amount,

        0,
    )


def apply_preset_rgb(
    img,
    preset,
):

    rgb = ensure_rgb(
        img
    )

    bgr = cv2.cvtColor(

        rgb,

        cv2.COLOR_RGB2BGR,
    )

    cfg = PRESETS[
        preset
    ]

    action = cfg[
        "action"
    ]

    if action == "none":

        out = bgr.copy()

    elif action == "gamma":

        out = apply_gamma(

            bgr,

            cfg[
                "gamma"
            ],
        )

    elif action == "clahe":

        out = apply_clahe(

            bgr,

            cfg[
                "clahe"
            ],
        )

    elif action == "gamma_clahe":

        out = apply_gamma(

            bgr,

            cfg[
                "gamma"
            ],
        )

        out = apply_clahe(

            out,

            cfg[
                "clahe"
            ],
        )

    elif action == "sharpen":

        out = apply_sharpen(

            bgr,

            cfg[
                "sharpen"
            ],
        )

    else:

        raise ValueError(
            f"未知 preset：{preset}"
        )

    return cv2.cvtColor(

        out,

        cv2.COLOR_BGR2RGB,
    )


# =========================================================
# YOLO
# =========================================================

def load_yolo():

    global yolo_model

    if yolo_model is not None:
        return

    if not YOLO_WEIGHTS.exists():

        raise FileNotFoundError(
            f"找不到 YOLO 權重：{YOLO_WEIGHTS}"
        )

    print()
    print("=" * 60)
    print("Loading YOLO")
    print(YOLO_WEIGHTS)
    print("=" * 60)

    yolo_model = YOLO(
        str(
            YOLO_WEIGHTS
        )
    )

    print(
        "YOLO READY"
    )


def run_yolo(img):

    if img is None:

        raise gr.Error(
            "請先產生增強後圖片"
        )

    load_yolo()

    rgb = ensure_rgb(
        img
    )

    bgr = cv2.cvtColor(

        rgb,

        cv2.COLOR_RGB2BGR,
    )

    results = (
        yolo_model.predict(

            source=bgr,

            conf=YOLO_CONF,

            imgsz=YOLO_IMGSZ,

            device=0,

            verbose=False,
        )
    )

    r = results[0]

    plotted_bgr = r.plot()

    plotted_rgb = cv2.cvtColor(

        plotted_bgr,

        cv2.COLOR_BGR2RGB,
    )

    if r.boxes is None:

        count = 0

    else:

        count = len(
            r.boxes
        )

    lines = [

        f"偵測數量：{count}",

        f"conf：{YOLO_CONF}",

        f"imgsz：{YOLO_IMGSZ}",
    ]

    if (
        r.boxes is not None
        and len(
            r.boxes
        ) > 0
    ):

        for i, box in enumerate(
            r.boxes
        ):

            cls_id = int(
                box.cls.item()
            )

            conf = float(
                box.conf.item()
            )

            name = r.names.get(
                cls_id,
                str(
                    cls_id
                ),
            )

            lines.append(

                f"{i + 1}. "
                f"{name} "
                f"{conf:.3f}"
            )

    return (

        plotted_rgb,

        "\n".join(
            lines
        ),
    )


# =========================================================
# UI CALLBACK
# =========================================================

def analyze_image(img):

    if img is None:

        raise gr.Error(
            "請先上傳圖片"
        )

    try:

        diagnosis, raw, stats = (
            ask_mllm(
                img
            )
        )

        preset, policy_reason = (
            choose_preset(
                diagnosis
            )
        )

        diagnosis_text = (
            json.dumps(

                diagnosis,

                ensure_ascii=False,

                indent=2,
            )
        )

        stats_text = (
            json.dumps(

                stats,

                ensure_ascii=False,

                indent=2,
            )
        )

        info = (

            f"目前圖片 Hash："
            f"{stats['image_sha256_short']}\n\n"

            f"建議 Preset：{preset}\n"

            f"名稱："
            f"{PRESETS[preset]['name']}\n\n"

            f"MLLM 原因："
            f"{diagnosis['reason']}\n\n"

            f"Policy 原因："
            f"{policy_reason}"
        )

        return (

            diagnosis_text,

            stats_text,

            raw,

            preset,

            info,
        )

    except Exception as e:

        raise gr.Error(
            f"MLLM 分析失敗：{repr(e)}"
        )


def preview_enhancement(
    img,
    preset,
):

    if img is None:

        raise gr.Error(
            "請先上傳圖片"
        )

    if preset not in PRESETS:

        raise gr.Error(
            "Preset 不正確"
        )

    out = apply_preset_rgb(

        img,

        preset,
    )

    cfg = PRESETS[
        preset
    ]

    text = (

        f"Preset：{preset}\n"

        f"名稱：{cfg['name']}\n"

        f"Action：{cfg['action']}\n"

        f"Gamma：{cfg['gamma']}\n"

        f"CLAHE：{cfg['clahe']}\n"

        f"Sharpen：{cfg['sharpen']}"
    )

    return (
        out,
        text,
    )


def full_pipeline(img):

    if img is None:

        raise gr.Error(
            "請先上傳圖片"
        )

    diagnosis, raw, stats = (
        ask_mllm(
            img
        )
    )

    preset, policy_reason = (
        choose_preset(
            diagnosis
        )
    )

    enhanced = apply_preset_rgb(

        img,

        preset,
    )

    yolo_img, yolo_text = (
        run_yolo(
            enhanced
        )
    )

    diagnosis_text = (
        json.dumps(

            diagnosis,

            ensure_ascii=False,

            indent=2,
        )
    )

    stats_text = (
        json.dumps(

            stats,

            ensure_ascii=False,

            indent=2,
        )
    )

    info = (

        f"目前圖片 Hash："
        f"{stats['image_sha256_short']}\n\n"

        f"建議 Preset：{preset}\n"

        f"名稱："
        f"{PRESETS[preset]['name']}\n\n"

        f"MLLM 原因："
        f"{diagnosis['reason']}\n\n"

        f"Policy 原因："
        f"{policy_reason}"
    )

    return (

        diagnosis_text,

        stats_text,

        raw,

        preset,

        info,

        enhanced,

        yolo_img,

        yolo_text,
    )


# =========================================================
# CLEAR OLD RESULTS WHEN IMAGE CHANGES
# =========================================================

def clear_results():

    return (

        "",

        "",

        "",

        "N",

        "",

        None,

        None,

        "",
    )


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="Night Agent V2"
) as demo:

    gr.Markdown(
        """
# 夜間 MLLM → OpenCV → YOLO26 V2

### 新流程

**圖片 → Qwen3-VL-8B 影像品質診斷 → Policy → OpenCV → YOLO26**

MLLM 不再直接選 Gamma / CLAHE / GC2。

MLLM 只分析：

- 黑暗程度
- 低對比
- 模糊
- 雜訊
- 過度曝光
- 小物體可見性

每次換圖片後，
舊的分析結果會自動清空。

每張圖片都有 Hash，
避免把上一張結果誤認成目前圖片。
"""
    )

    with gr.Row():

        input_image = gr.Image(

            label="原始圖片",

            type="numpy",
        )

        enhanced_image = gr.Image(

            label="OpenCV 增強後",

            type="numpy",
        )

        yolo_image = gr.Image(

            label="YOLO26 結果",

            type="numpy",
        )

    analyze_btn = gr.Button(

        "① Qwen3-VL-8B 分析圖片",

        variant="primary",
    )

    with gr.Row():

        diagnosis_text = gr.Textbox(

            label="MLLM 結構化診斷",

            lines=12,
        )

        stats_text = gr.Textbox(

            label="電腦影像統計",

            lines=12,
        )

    raw_output = gr.Textbox(

        label="MLLM 原始輸出（Debug 用）",

        lines=10,
    )

    policy_info = gr.Textbox(

        label="診斷與 Policy 說明",

        lines=9,
    )

    preset_dropdown = gr.Dropdown(

        choices=list(
            PRESETS.keys()
        ),

        value="N",

        label="最終 Preset（可人工覆蓋）",
    )

    preview_btn = gr.Button(
        "② 套用 OpenCV / 預覽"
    )

    preset_info = gr.Textbox(

        label="實際 OpenCV 參數",

        lines=6,
    )

    yolo_btn = gr.Button(

        "③ 將增強後圖片送入 YOLO26",

        variant="primary",
    )

    yolo_info = gr.Textbox(

        label="YOLO 偵測資訊",

        lines=8,
    )

    gr.Markdown(
        "---\n### 快速 Demo"
    )

    full_btn = gr.Button(

        "🚀 一鍵：MLLM → OpenCV → YOLO26",

        variant="primary",
    )


    # -----------------------------------------------------
    # 換圖片後自動清掉上一張結果
    # -----------------------------------------------------

    input_image.change(

        fn=clear_results,

        inputs=[],

        outputs=[

            diagnosis_text,

            stats_text,

            raw_output,

            preset_dropdown,

            policy_info,

            enhanced_image,

            yolo_image,

            yolo_info,
        ],
    )


    # -----------------------------------------------------
    # MLLM
    # -----------------------------------------------------

    analyze_btn.click(

        fn=analyze_image,

        inputs=[
            input_image
        ],

        outputs=[

            diagnosis_text,

            stats_text,

            raw_output,

            preset_dropdown,

            policy_info,
        ],
    )


    # -----------------------------------------------------
    # OPENCV
    # -----------------------------------------------------

    preview_btn.click(

        fn=preview_enhancement,

        inputs=[

            input_image,

            preset_dropdown,
        ],

        outputs=[

            enhanced_image,

            preset_info,
        ],
    )


    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

    yolo_btn.click(

        fn=run_yolo,

        inputs=[
            enhanced_image
        ],

        outputs=[

            yolo_image,

            yolo_info,
        ],
    )


    # -----------------------------------------------------
    # FULL PIPELINE
    # -----------------------------------------------------

    full_btn.click(

        fn=full_pipeline,

        inputs=[
            input_image
        ],

        outputs=[

            diagnosis_text,

            stats_text,

            raw_output,

            preset_dropdown,

            policy_info,

            enhanced_image,

            yolo_image,

            yolo_info,
        ],
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    demo.launch(

        server_name="127.0.0.1",

        server_port=7861,

        inbrowser=True,
    )