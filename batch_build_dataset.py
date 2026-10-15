from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)


# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path(
    r"C:\Users\user\Desktop\專題 001\night_data_v3_split"
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ---------------------------------------------------------
# Reproducible enhancement presets
# MLLM only chooses preset ID.
# It does NOT invent arbitrary parameters.
# ---------------------------------------------------------

PRESETS = {
    "N": {
        "action": "none",
        "gamma": 1.0,
        "clahe": 0.0,
        "sharpen": 0.0,
    },

    "G1": {
        "action": "gamma",
        "gamma": 0.65,
        "clahe": 0.0,
        "sharpen": 0.0,
    },

    "G2": {
        "action": "gamma",
        "gamma": 0.80,
        "clahe": 0.0,
        "sharpen": 0.0,
    },

    "C1": {
        "action": "clahe",
        "gamma": 1.0,
        "clahe": 1.5,
        "sharpen": 0.0,
    },

    "C2": {
        "action": "clahe",
        "gamma": 1.0,
        "clahe": 2.5,
        "sharpen": 0.0,
    },

    "GC1": {
        "action": "gamma_clahe",
        "gamma": 0.65,
        "clahe": 2.0,
        "sharpen": 0.0,
    },

    "GC2": {
        "action": "gamma_clahe",
        "gamma": 0.80,
        "clahe": 2.0,
        "sharpen": 0.0,
    },

    "S1": {
        "action": "sharpen",
        "gamma": 1.0,
        "clahe": 0.0,
        "sharpen": 0.50,
    },

    "S2": {
        "action": "sharpen",
        "gamma": 1.0,
        "clahe": 0.0,
        "sharpen": 1.00,
    },
}


PROMPT = """
You are an adaptive image-preprocessing selector.

The downstream task is nighttime SMALL PERSON detection using YOLO26n.

You are NOT the person detector.
You are NOT allowed to modify bounding boxes.

Choose exactly ONE preprocessing preset for the current image.

Goal:
Improve visibility of small people while avoiding excessive enhancement,
noise amplification, halos, and false-positive background texture.

Available presets:

N   = no change
G1  = Gamma 0.65, strong brightness increase
G2  = Gamma 0.80, mild brightness increase
C1  = CLAHE 1.5, mild local contrast
C2  = CLAHE 2.5, stronger local contrast
GC1 = Gamma 0.65 + CLAHE 2.0
GC2 = Gamma 0.80 + CLAHE 2.0
S1  = mild sharpen 0.50
S2  = stronger sharpen 1.00

Rules:
- Do not always brighten nighttime images.
- If the image is already clear, choose N.
- Avoid over-enhancement.
- Strong texture enhancement can increase YOLO false positives.
- Small people must remain visually natural.
- Use both the image and the supplied numerical statistics.

OUTPUT JSON ONLY:

{"preset":"GC2"}
"""


# =========================================================
# IMAGE IO
# =========================================================

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
            "Cannot read: " + str(path)
        )

    return img


def save_jpg(path, img):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:
        raise RuntimeError(
            "Cannot encode image"
        )

    buf.tofile(str(path))


# =========================================================
# IMAGE STATS
# =========================================================

def stats(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "brightness":
            round(float(gray.mean()), 2),

        "contrast_std":
            round(float(gray.std()), 2),

        "dark_ratio":
            round(
                float(np.mean(gray < 50)),
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
            ),
    }


# =========================================================
# ENHANCEMENT
# =========================================================

def gamma_enhance(img, gamma):

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ])

    table = np.clip(
        table,
        0,
        255
    ).astype(np.uint8)

    return cv2.LUT(
        img,
        table
    )


def clahe_enhance(img, clip):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip),
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )


def sharpen_enhance(img, strength):

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.0
    )

    return cv2.addWeighted(
        img,
        1.0 + strength,
        blur,
        -strength,
        0
    )


def apply_preset(img, preset_id):

    p = PRESETS[preset_id]

    action = p["action"]

    if action == "none":
        return img.copy()

    if action == "gamma":
        return gamma_enhance(
            img,
            p["gamma"]
        )

    if action == "clahe":
        return clahe_enhance(
            img,
            p["clahe"]
        )

    if action == "gamma_clahe":

        x = gamma_enhance(
            img,
            p["gamma"]
        )

        return clahe_enhance(
            x,
            p["clahe"]
        )

    if action == "sharpen":

        return sharpen_enhance(
            img,
            p["sharpen"]
        )

    raise RuntimeError(
        "Unknown action"
    )


# =========================================================
# DATASET
# =========================================================

def find_split(root, name):

    aliases = {
        "train": [
            "train"
        ],

        "valid": [
            "val",
            "valid",
            "validation"
        ],
    }

    for alias in aliases[name]:

        image_dir = (
            root
            / "images"
            / alias
        )

        label_dir = (
            root
            / "labels"
            / alias
        )

        if (
            image_dir.exists()
            and label_dir.exists()
        ):

            print(
                "Found split:",
                name,
                "->",
                image_dir
            )

            return (
                image_dir,
                label_dir
            )

    raise RuntimeError(
        "Cannot find split: "
        + name
    )

def list_images(folder):

    result = []

    for path in folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTS
        ):

            result.append(path)

    return sorted(result)


# =========================================================
# MLLM
# =========================================================

def load_model():

    print()
    print("=" * 60)
    print("Loading MLLM")
    print(MODEL_ID)
    print("4-bit NF4")
    print("=" * 60)
    print()

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_ID
        )
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=quant,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    return model, processor


def resize_for_mllm(img):

    h, w = img.shape[:2]

    max_side = 896

    scale = min(
        1.0,
        max_side / max(h, w)
    )

    if scale >= 1:
        return img

    return cv2.resize(
        img,
        (
            int(w * scale),
            int(h * scale)
        ),
        interpolation=cv2.INTER_AREA
    )


def parse_preset(raw):

    match = re.search(
        r"\{.*?\}",
        raw,
        re.S
    )

    if not match:
        raise ValueError(
            "No JSON"
        )

    obj = json.loads(
        match.group(0)
    )

    preset = str(
        obj.get(
            "preset",
            ""
        )
    ).strip().upper()

    if preset not in PRESETS:
        raise ValueError(
            "Invalid preset: " + preset
        )

    return preset


def decide(
    model,
    processor,
    img,
    image_stats
):

    small = resize_for_mllm(
        img
    )

    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )

    pil = Image.fromarray(
        rgb
    )

    user_text = (
        PROMPT
        + "\nIMAGE_STATS:\n"
        + json.dumps(
            image_stats,
            indent=2
        )
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil
                },
                {
                    "type": "text",
                    "text": user_text
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        model.device
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=24,
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

    raw = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return (
        parse_preset(raw),
        raw
    )


# =========================================================
# CSV RESUME
# =========================================================

def read_completed(csv_path):

    completed = set()

    if not csv_path.exists():
        return completed

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row.get("status") == "ok":

                completed.add(
                    (
                        row["split"],
                        row["source_rel"]
                    )
                )

    return completed


def write_yaml(
    output_root
):

    text = f"""path: '{output_root}'
train: train/images
val: valid/images

names:
  0: person
  1: dog
  2: car
  3: bicycle
"""

    (
        output_root
        / "data.yaml"
    ).write_text(
        text,
        encoding="utf-8"
    )


# =========================================================
# MAIN PROCESS
# =========================================================

def build_dataset(
    strategy,
    output_root
):

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_root
        / "decisions.csv"
    )

    completed = read_completed(
        csv_path
    )

    need_header = (
        not csv_path.exists()
    )

    model = None
    processor = None

    if strategy == "mllm":

        model, processor = (
            load_model()
        )

    fields = [
        "split",
        "source_rel",
        "output_rel",
        "preset",
        "action",
        "gamma",
        "clahe",
        "sharpen",
        "brightness",
        "contrast_std",
        "dark_ratio",
        "sharpness",
        "seconds",
        "status",
        "raw_mllm",
    ]

    counts = {
        key: 0
        for key in PRESETS
    }

    with csv_path.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields
        )

        if need_header:
            writer.writeheader()
            csv_file.flush()

        for split in [
            "train",
            "valid"
        ]:

            image_dir, label_dir = (
                find_split(
                    DATASET_ROOT,
                    split
                )
            )

            images = list_images(
                image_dir
            )

            print()
            print("=" * 60)
            print(
                split,
                len(images),
                "images"
            )
            print("=" * 60)

            progress = tqdm(
                images,
                desc=split
            )

            for image_path in progress:

                rel = image_path.relative_to(
                    image_dir
                )

                rel_str = str(rel)

                key = (
                    split,
                    rel_str
                )

                if key in completed:
                    continue

                started = time.time()

                status = "ok"
                raw = ""

                try:

                    img = read_image(
                        image_path
                    )

                    s = stats(
                        img
                    )

                    if strategy == "fixed":

                        preset = "GC2"

                    else:

                        preset = None

                        last_error = None

                        for attempt in range(3):

                            try:

                                preset, raw = decide(
                                    model,
                                    processor,
                                    img,
                                    s
                                )

                                break

                            except Exception as e:

                                last_error = e

                                raw = repr(e)

                                time.sleep(0.5)

                        if preset is None:

                            preset = "N"

                            status = (
                                "fallback_none"
                            )

                    processed = (
                        apply_preset(
                            img,
                            preset
                        )
                    )

                    preset_cfg = (
                        PRESETS[preset]
                    )

                    # -------------------------------
                    # Output image
                    # -------------------------------

                    out_image_rel = (
                        Path(split)
                        / "images"
                        / rel
                    )

                    if preset == "N":

                        out_image = (
                            output_root
                            / out_image_rel
                        )

                        out_image.parent.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        shutil.copy2(
                            image_path,
                            out_image
                        )

                    else:

                        out_image_rel = (
                            out_image_rel
                            .with_suffix(".jpg")
                        )

                        out_image = (
                            output_root
                            / out_image_rel
                        )

                        save_jpg(
                            out_image,
                            processed
                        )

                    # -------------------------------
                    # Label
                    # -------------------------------

                    label_source = (
                        label_dir
                        / rel
                    ).with_suffix(
                        ".txt"
                    )

                    label_target = (
                        output_root
                        / split
                        / "labels"
                        / rel
                    ).with_suffix(
                        ".txt"
                    )

                    if label_source.exists():

                        label_target.parent.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        shutil.copy2(
                            label_source,
                            label_target
                        )

                    elapsed = (
                        time.time()
                        - started
                    )

                    counts[
                        preset
                    ] += 1

                    row = {
                        "split":
                            split,

                        "source_rel":
                            rel_str,

                        "output_rel":
                            str(out_image_rel),

                        "preset":
                            preset,

                        "action":
                            preset_cfg[
                                "action"
                            ],

                        "gamma":
                            preset_cfg[
                                "gamma"
                            ],

                        "clahe":
                            preset_cfg[
                                "clahe"
                            ],

                        "sharpen":
                            preset_cfg[
                                "sharpen"
                            ],

                        "brightness":
                            s["brightness"],

                        "contrast_std":
                            s["contrast_std"],

                        "dark_ratio":
                            s["dark_ratio"],

                        "sharpness":
                            s["sharpness"],

                        "seconds":
                            round(
                                elapsed,
                                3
                            ),

                        "status":
                            status,

                        "raw_mllm":
                            raw.replace(
                                "\n",
                                " "
                            )[:500],
                    }

                    writer.writerow(
                        row
                    )

                    csv_file.flush()

                    completed.add(
                        key
                    )

                    progress.set_postfix(
                        preset=preset,
                        sec=f"{elapsed:.1f}"
                    )

                except Exception as e:

                    elapsed = (
                        time.time()
                        - started
                    )

                    writer.writerow({
                        "split":
                            split,

                        "source_rel":
                            rel_str,

                        "output_rel":
                            "",

                        "preset":
                            "",

                        "action":
                            "",

                        "gamma":
                            "",

                        "clahe":
                            "",

                        "sharpen":
                            "",

                        "brightness":
                            "",

                        "contrast_std":
                            "",

                        "dark_ratio":
                            "",

                        "sharpness":
                            "",

                        "seconds":
                            round(
                                elapsed,
                                3
                            ),

                        "status":
                            "error",

                        "raw_mllm":
                            repr(e)[:500],
                    })

                    csv_file.flush()

                    print()
                    print(
                        "ERROR:",
                        rel_str,
                        repr(e)
                    )

    write_yaml(
        output_root
    )

    summary = {
        "strategy":
            strategy,

        "model":
            (
                MODEL_ID
                if strategy == "mllm"
                else None
            ),

        "preset_counts":
            counts,

        "dataset_root":
            str(DATASET_ROOT),

        "output_root":
            str(output_root),
    }

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("DONE")
    print(output_root)
    print("=" * 60)
    print()
    print(
        json.dumps(
            counts,
            indent=2
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        choices=[
            "fixed",
            "mllm",
        ],
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    build_dataset(
        strategy=args.strategy,
        output_root=Path(
            args.output
        ),
    )


if __name__ == "__main__":
    main()
