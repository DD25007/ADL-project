from datasets import (
    Dataset,
    Features,
    Value,
    Image as HfImage,
)
from PIL import Image as PilImage
import numpy as np
import os

import json
from scipy import ndimage  # to find connected components


base_dir = "data/Dataset_BUSI_with_GT/"

records = []
for label in ["benign", "malignant"]:
    folder = os.path.join(base_dir, label)
    for file in os.listdir(folder):
        if not file.endswith(".png") or "mask" in file:
            continue

        img_path = os.path.join(folder, file)

        # Find all masks for this image
        base_name = file.replace(".png", "")
        mask_files = [
            f for f in os.listdir(folder) if f.startswith(base_name + "_mask")
        ]

        if mask_files:
            masks = []
            for m in mask_files:
                mask_img = np.array(PilImage.open(os.path.join(folder, m)).convert("L"))
                masks.append(mask_img > 0)

            merged_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            conditioning_image = PilImage.fromarray(merged_mask)
        else:
            conditioning_image_path = None

        records.append(
            {
                "image": img_path,  # just the path
                "conditioning_image": conditioning_image,  # PIL Image or None
                "caption": label,
            }
        )

# Define features so Hugging Face knows how to decode them
features = Features(
    {"image": HfImage(), "conditioning_image": HfImage(), "caption": Value("string")}
)

kaggle_dataset = Dataset.from_list(records, features=features)

# Save the whole merged dataset's only image
root_dir = "data/busi/"

# create per-label subfolders and save using global index so names match JSON builder
for label in ["benign", "malignant"]:
    os.makedirs(os.path.join(root_dir, label), exist_ok=True)

# Save every image with global index and same naming convention used in JSON
for global_idx in range(len(kaggle_dataset)):
    example = kaggle_dataset[global_idx]  # forces HFImage to decode
    label = "benign" if "benign" in example["caption"].lower() else "malignant"
    image = example["image"]  # PIL Image
    save_name = f"{label}_img_{global_idx:04d}.png"
    save_path = os.path.join(root_dir, label, save_name)
    image.save(save_path)

print(f"Saved {len(kaggle_dataset)} images under {root_dir}")


def masks_to_coco_json(dataset, output_json="data/busi_bboxes.json"):
    categories = [
        {"id": 1, "name": "benign", "encode_name": "benign"},
        {"id": 2, "name": "malignant", "encode_name": "malignant"},
    ]

    images = []
    annotations = []
    ann_id = 1

    for idx, example in enumerate(
        dataset
    ):  # idx is global index => matches saved file names
        img = example["image"]
        mask = example["conditioning_image"]
        label = "benign" if "benign" in example["caption"].lower() else "malignant"

        # Decode mask if HFImage dict or PIL
        if isinstance(mask, dict):
            mask = mask["array"]
        elif isinstance(mask, PilImage.Image):
            mask = np.array(mask)

        # get width/height
        if isinstance(img, dict):
            h, w = img["array"].shape[:2]
        else:
            w, h = img.size

        # file_name must match saved file on disk: "<label>/<label>_img_<global_idx:04d>.png"
        file_name = f"{label}/{label}_img_{idx:04d}.png"

        images.append(
            {
                "file_name": file_name,
                "height": h,
                "width": w,
                "id": idx + 1,
            }
        )

        # connected components -> multiple lesions
        labeled_mask, num_features = ndimage.label(mask > 0)
        for lesion_idx in range(1, num_features + 1):
            ys, xs = np.where(labeled_mask == lesion_idx)
            if len(xs) == 0 or len(ys) == 0:
                continue
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            box_w, box_h = xmax - xmin, ymax - ymin

            category_id = 1 if label == "benign" else 2

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": category_id,
                    "bbox": [int(xmin), int(ymin), int(box_w), int(box_h)],
                    "area": int(box_w * box_h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_dict = {"categories": categories, "images": images, "annotations": annotations}

    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"✅ COCO JSON saved: {output_json}")
    return coco_dict


coco_json = masks_to_coco_json(kaggle_dataset, output_json="data/busi_bboxes.json")
