from pathlib import Path
import hashlib
import json

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


# =========================================================
# PRESETS
# =========================================================

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
# IMAGE UTILS
# =========================================================

def ensure_rgb(img):
    if img is None:
        raise ValueError("請先上傳圖片")

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

    return arr.astype(np.uint8)


def image_hash(img):
    rgb = ensure_rgb(img)

    return hashlib.sha256(
        rgb.tobytes()
    ).hexdigest()[:16]


# =========================================================
# COMPUTER VISION STATS
# =========================================================

def calculate_stats(img):
    """
    V3 核心：
    可精確量測的低階影像品質，不再交給 MLLM 猜。
    """

    rgb = ensure_rgb(img)

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
        np.mean(gray < 50)
    )

    overexposure_ratio = float(
        np.mean(gray > 230)
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
                gray.astype(np.float32)
                -
                median.astype(np.float32)
            )
        )
    )

    return {
        "brightness": round(
            brightness,
            2,
        ),

        "contrast_std": round(
            contrast_std,
            2,
        ),

        "dark_ratio": round(
            dark_ratio,
            4,
        ),

        "overexposure_ratio": round(
            overexposure_ratio,
            4,
        ),

        "sharpness": round(
            sharpness,
            2,
        ),

        "noise_estimate": round(
            noise_estimate,
            2,
        ),

        "image_sha256_short":
            image_hash(rgb),
    }


# =========================================================
# LOAD MLLM
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

    print("MLLM READY")


# =========================================================
# MLLM JSON
# =========================================================

def parse_semantic_json(raw):
    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "MLLM 沒有輸出 JSON"
        )

    obj = json.loads(
        raw[start:end + 1]
    )

    required = [
        "small_object_visibility",
        "subject_background_separation",
        "occlusion",
        "backlight",
        "reason",
    ]

    for key in required:
        if key not in obj:
            raise ValueError(
                f"缺少欄位：{key}"
            )

    numeric_keys = [
        "small_object_visibility",
        "subject_background_separation",
        "occlusion",
        "backlight",
    ]

    for key in numeric_keys:
        obj[key] = int(
            float(obj[key])
        )

        obj[key] = max(
            0,
            min(
                100,
                obj[key],
            ),
        )

    obj["reason"] = str(
        obj["reason"]
    )

    return obj


# =========================================================
# MLLM SEMANTIC ANALYSIS
# =========================================================

def ask_mllm(img):
    """
    V3:
    MLLM 不再估 brightness / contrast / noise / blur。

    它只負責需要語意理解的內容。
    """

    load_mllm()

    rgb = ensure_rgb(img)

    stats = calculate_stats(
        rgb
    )

    pil = Image.fromarray(
        rgb
    )

    prompt = """
You are the semantic visual-analysis module in a nighttime
small-object detection research system.

IMPORTANT:

Do NOT estimate:
- brightness
- darkness
- image noise level
- contrast
- sharpness
- blur amount
- exposure percentage

Those quantities are measured separately by computer vision.

Do NOT choose any enhancement method.

Do NOT choose:
Gamma
CLAHE
Sharpen
G1
G2
C1
C2
GC1
GC2
S1
S2

Your task is ONLY to analyze semantic visibility for object detection.

Score from 0 to 100:

small_object_visibility:
0 = tiny relevant objects are almost impossible to identify
100 = tiny relevant objects are visually easy to identify

subject_background_separation:
0 = objects visually blend into the background
100 = objects are clearly separated from the background

occlusion:
0 = relevant objects are not occluded
100 = relevant objects are heavily occluded

backlight:
0 = no meaningful backlight problem
100 = severe backlighting makes object recognition difficult

reason:
ONE short Traditional Chinese sentence explaining the major
semantic visibility problem.

Return JSON ONLY.

Exact schema:

{
  "small_object_visibility": 0,
  "subject_background_separation": 0,
  "occlusion": 0,
  "backlight": 0,
  "reason": "..."
}
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
        output = mllm_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            use_cache=True,
        )

    input_len = (
        inputs["input_ids"]
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

    semantic = parse_semantic_json(
        raw
    )

    return (
        semantic,
        raw,
        stats,
    )


# =========================================================
# HYBRID POLICY
# =========================================================

def choose_preset(
    stats,
    semantic,
):
    """
    V3 核心決策：

    OpenCV stats：
    決定到底是真的暗、低對比、過曝、模糊。

    MLLM semantic：
    決定小物件是否真的難辨識，以及背景分離狀況。

    注意：
    這些 threshold 是目前的研究起始值，
    後續要用 Validation 調整。
    """

    brightness = stats[
        "brightness"
    ]

    contrast = stats[
        "contrast_std"
    ]

    dark_ratio = stats[
        "dark_ratio"
    ]

    over_ratio = stats[
        "overexposure_ratio"
    ]

    sharpness = stats[
        "sharpness"
    ]

    noise = stats[
        "noise_estimate"
    ]

    visibility = semantic[
        "small_object_visibility"
    ]

    separation = semantic[
        "subject_background_separation"
    ]

    backlight = semantic[
        "backlight"
    ]


    # =====================================================
    # 1. Low-level state
    # =====================================================

    very_dark = (
        brightness < 55
        or dark_ratio > 0.55
    )

    moderately_dark = (
        brightness < 80
        or dark_ratio > 0.30
    )

    very_low_contrast = (
        contrast < 20
    )

    low_contrast = (
        contrast < 30
    )

    overexposed = (
        over_ratio > 0.08
    )

    very_blurry = (
        sharpness < 70
    )

    blurry = (
        sharpness < 120
    )

    high_noise = (
        noise > 8
    )


    # =====================================================
    # 2. Semantic difficulty
    # =====================================================

    semantic_hard = (
        visibility < 40
        or separation < 40
        or backlight > 70
    )

    semantic_medium = (
        visibility < 65
        or separation < 60
        or backlight > 45
    )


    # =====================================================
    # 3. Overexposure
    # =====================================================

    if overexposed:

        if low_contrast:
            return (
                "C1",
                "電腦量測顯示有過曝風險，"
                "避免再提亮，只使用輕度 CLAHE。",
            )

        return (
            "N",
            "電腦量測顯示已有過曝區域，"
            "因此不增加亮度。",
        )


    # =====================================================
    # 4. Very dark + contrast
    # =====================================================

    if (
        very_dark
        and low_contrast
    ):

        if semantic_hard:
            return (
                "GC1",
                "影像非常暗且低對比，"
                "MLLM 也判斷小物件辨識困難，"
                "使用較強 Gamma + CLAHE。",
            )

        return (
            "GC2",
            "影像非常暗且低對比，"
            "但語意可見性尚可，"
            "先使用較溫和 Gamma + CLAHE。",
        )


    # =====================================================
    # 5. Moderate dark + contrast
    # =====================================================

    if (
        moderately_dark
        and low_contrast
    ):

        if semantic_hard:
            return (
                "GC1",
                "影像偏暗且低對比，"
                "同時小物件辨識困難，"
                "提高增強強度。",
            )

        return (
            "GC2",
            "影像偏暗且對比不足，"
            "使用溫和 Gamma + CLAHE。",
        )


    # =====================================================
    # 6. Darkness only
    # =====================================================

    if very_dark:

        if semantic_hard:
            return (
                "G1",
                "主要問題為亮度不足，"
                "且小物件語意可見性差，"
                "使用較強 Gamma。",
            )

        return (
            "G2",
            "主要問題為亮度不足，"
            "使用溫和 Gamma。",
        )


    if moderately_dark:

        if semantic_hard:
            return (
                "G2",
                "影像稍暗且小物件辨識較困難，"
                "使用溫和 Gamma。",
            )


    # =====================================================
    # 7. Contrast only
    # =====================================================

    if very_low_contrast:

        if semantic_hard:
            return (
                "C2",
                "亮度尚可，但電腦量測顯示對比非常低，"
                "且小物件與背景分離差。",
            )

        return (
            "C1",
            "亮度尚可但對比不足，"
            "使用輕度 CLAHE。",
        )


    if low_contrast:

        if semantic_medium:
            return (
                "C1",
                "畫面對比偏低，"
                "且 MLLM 判斷物體分離度有限。",
            )


    # =====================================================
    # 8. Blur
    # =====================================================

    if (
        very_blurry
        and not high_noise
    ):

        return (
            "S2",
            "電腦量測顯示影像非常模糊，"
            "且雜訊不高，使用較強 Sharpen。",
        )


    if (
        blurry
        and not high_noise
    ):

        if semantic_medium:
            return (
                "S1",
                "電腦量測顯示影像稍模糊，"
                "且小物件辨識度有限，"
                "使用輕度 Sharpen。",
            )


    # =====================================================
    # 9. Semantic difficulty alone
    # =====================================================

    if semantic_hard:
        return (
            "N",
            "MLLM 判斷小物件辨識困難，"
            "但低階影像統計沒有顯示可由"
            "Gamma、CLAHE 或 Sharpen 明確改善的問題；"
            "因此不強制增強。",
        )


    # =====================================================
    # 10. Normal
    # =====================================================

    return (
        "N",
        "低階影像統計正常，"
        "MLLM 語意可見性也沒有明顯異常，"
        "保留原圖。",
    )


# =========================================================
# OPENCV ENHANCEMENT
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
            ) * 255
            for i in range(256)
        ],
        dtype=np.uint8,
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
        clipLimit=float(clip),
        tileGridSize=(8, 8),
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
        (0, 0),
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
            cfg["gamma"],
        )

    elif action == "clahe":
        out = apply_clahe(
            bgr,
            cfg["clahe"],
        )

    elif action == "gamma_clahe":
        out = apply_gamma(
            bgr,
            cfg["gamma"],
        )

        out = apply_clahe(
            out,
            cfg["clahe"],
        )

    elif action == "sharpen":
        out = apply_sharpen(
            bgr,
            cfg["sharpen"],
        )

    else:
        raise ValueError(
            f"未知 Preset：{preset}"
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
        str(YOLO_WEIGHTS)
    )

    print("YOLO READY")


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

    results = yolo_model.predict(
        source=bgr,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device=0,
        verbose=False,
    )

    r = results[0]

    plotted_bgr = r.plot()

    plotted_rgb = cv2.cvtColor(
        plotted_bgr,
        cv2.COLOR_BGR2RGB,
    )

    count = (
        0
        if r.boxes is None
        else len(r.boxes)
    )

    lines = [
        f"偵測數量：{count}",
        f"conf：{YOLO_CONF}",
        f"imgsz：{YOLO_IMGSZ}",
    ]

    if (
        r.boxes is not None
        and len(r.boxes) > 0
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
                str(cls_id),
            )

            lines.append(
                f"{i + 1}. "
                f"{name} "
                f"{conf:.3f}"
            )

    return (
        plotted_rgb,
        "\n".join(lines),
    )


# =========================================================
# ANALYZE
# =========================================================

def analyze_image(img):
    if img is None:
        raise gr.Error(
            "請先上傳圖片"
        )

    try:
        semantic, raw, stats = (
            ask_mllm(img)
        )

        preset, policy_reason = (
            choose_preset(
                stats,
                semantic,
            )
        )

        semantic_text = json.dumps(
            semantic,
            ensure_ascii=False,
            indent=2,
        )

        stats_text = json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )

        info = (
            f"目前圖片 Hash："
            f"{stats['image_sha256_short']}\n\n"

            f"建議 Preset：{preset}\n"

            f"名稱："
            f"{PRESETS[preset]['name']}\n\n"

            f"MLLM 語意判斷：\n"
            f"{semantic['reason']}\n\n"

            f"Hybrid Policy：\n"
            f"{policy_reason}"
        )

        return (
            semantic_text,
            stats_text,
            raw,
            preset,
            info,
        )

    except Exception as e:
        raise gr.Error(
            f"分析失敗：{repr(e)}"
        )


# =========================================================
# PREVIEW
# =========================================================

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


# =========================================================
# FULL PIPELINE
# =========================================================

def full_pipeline(img):
    if img is None:
        raise gr.Error(
            "請先上傳圖片"
        )

    semantic, raw, stats = (
        ask_mllm(img)
    )

    preset, policy_reason = (
        choose_preset(
            stats,
            semantic,
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

    semantic_text = json.dumps(
        semantic,
        ensure_ascii=False,
        indent=2,
    )

    stats_text = json.dumps(
        stats,
        ensure_ascii=False,
        indent=2,
    )

    info = (
        f"目前圖片 Hash："
        f"{stats['image_sha256_short']}\n\n"

        f"建議 Preset：{preset}\n"

        f"名稱："
        f"{PRESETS[preset]['name']}\n\n"

        f"MLLM 語意判斷：\n"
        f"{semantic['reason']}\n\n"

        f"Hybrid Policy：\n"
        f"{policy_reason}"
    )

    return (
        semantic_text,
        stats_text,
        raw,
        preset,
        info,
        enhanced,
        yolo_img,
        yolo_text,
    )


# =========================================================
# CLEAR
# =========================================================

def clear_results():
    return (
        "",
        "",
        "",
        "N",
        "",
        None,
        "",
        None,
        "",
    )


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="Night Agent V3 Hybrid"
) as demo:

    gr.Markdown(
        """
# 夜間 MLLM × Computer Vision → OpenCV → YOLO26 V3

## Hybrid Decision

### 電腦負責精確量測
- Brightness
- Contrast
- Dark Ratio
- Overexposure
- Sharpness
- Noise

### Qwen3-VL-8B 只負責語意理解
- 小物件可見性
- 物體 / 背景分離程度
- 遮擋
- 逆光
- 語意原因

### 最後
**CV Stats + MLLM Semantic → Hybrid Policy → OpenCV → YOLO26**

這版不再讓 MLLM 猜 Brightness / Noise / Contrast。
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
        "① Hybrid 分析圖片",
        variant="primary",
    )


    with gr.Row():

        semantic_text = gr.Textbox(
            label="Qwen3-VL 語意診斷",
            lines=11,
        )

        stats_text = gr.Textbox(
            label="Computer Vision 精確統計",
            lines=11,
        )


    raw_output = gr.Textbox(
        label="MLLM 原始輸出（Debug）",
        lines=8,
    )


    policy_info = gr.Textbox(
        label="Hybrid Policy 決策",
        lines=10,
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
        label="OpenCV 實際參數",
        lines=6,
    )


    yolo_btn = gr.Button(
        "③ 將增強圖送入 YOLO26",
        variant="primary",
    )


    yolo_info = gr.Textbox(
        label="YOLO 偵測資訊",
        lines=8,
    )


    gr.Markdown(
        "---\n## 快速 Demo"
    )


    full_btn = gr.Button(
        "🚀 一鍵：Hybrid → OpenCV → YOLO26",
        variant="primary",
    )


    # =====================================================
    # IMAGE CHANGE
    # =====================================================

    input_image.change(
        fn=clear_results,
        inputs=[],
        outputs=[
            semantic_text,
            stats_text,
            raw_output,
            preset_dropdown,
            policy_info,
            enhanced_image,
            preset_info,
            yolo_image,
            yolo_info,
        ],
    )


    # =====================================================
    # ANALYZE
    # =====================================================

    analyze_btn.click(
        fn=analyze_image,
        inputs=[
            input_image
        ],
        outputs=[
            semantic_text,
            stats_text,
            raw_output,
            preset_dropdown,
            policy_info,
        ],
    )


    # =====================================================
    # OPENCV
    # =====================================================

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


    # =====================================================
    # YOLO
    # =====================================================

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


    # =====================================================
    # FULL
    # =====================================================

    full_btn.click(
        fn=full_pipeline,
        inputs=[
            input_image
        ],
        outputs=[
            semantic_text,
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
        server_port=7862,
        inbrowser=True,
    )