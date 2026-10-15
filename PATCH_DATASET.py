from pathlib import Path
import re

p = Path("batch_build_dataset.py")
s = p.read_text(encoding="utf-8-sig")

# 1. Qwen3-VL model class
s = s.replace(
    "AutoModelForMultimodalLM,",
    "Qwen3VLForConditionalGeneration,"
)

s = s.replace(
    "AutoModelForMultimodalLM\n        .from_pretrained(",
    "Qwen3VLForConditionalGeneration\n        .from_pretrained("
)

# 2. Replace find_split() for actual dataset structure:
#    images/train, images/val
#    labels/train, labels/val

start = s.index("def find_split(root, name):")
end = s.index("\ndef list_images(", start)

new_func = r'''def find_split(root, name):

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

'''

s = s[:start] + new_func + s[end+1:]

p.write_text(
    s,
    encoding="utf-8"
)

print("PATCH DONE")
