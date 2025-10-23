"""
Complete Pipeline Using Your Dataset Loading Code
Integrates your HuggingFace dataset approach with YOLO training
"""

import os
import yaml
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    Image as HfImage,
    load_dataset,
    concatenate_datasets,
)
from PIL import Image as PilImage

try:
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    print("Install: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("Install: pip install ultralytics")
    YOLO_AVAILABLE = False


# ===============================================
# STEP 1: LOAD DATA (USING YOUR CODE)
# ===============================================


def load_busi_with_merged_masks():
    """Your original code - loads BUSI with merged masks"""
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA WITH YOUR METHOD")
    print("=" * 60)

    base_dir = "../data/Dataset_BUSI_with_GT/"

    if not os.path.exists(base_dir):
        print(f"Error: Dataset not found at {base_dir}")
        return None

    records = []
    for label in ["benign", "malignant", "normal"]:
        folder = os.path.join(base_dir, label)

        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue

        print(f"Processing {label} folder...")
        count = 0

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
                    mask_img = np.array(
                        PilImage.open(os.path.join(folder, m)).convert("L")
                    )
                    masks.append(mask_img > 0)

                # Merge all masks
                merged_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                conditioning_image = PilImage.fromarray(merged_mask)
            else:
                conditioning_image = None

            records.append(
                {
                    "image": img_path,
                    "conditioning_image": conditioning_image,
                    "caption": label,
                }
            )
            count += 1

        print(f"  Loaded {count} {label} images")

    # Define features
    features = Features(
        {
            "image": HfImage(),
            "conditioning_image": HfImage(),
            "caption": Value("string"),
        }
    )

    kaggle_dataset = Dataset.from_list(records, features=features)
    print(f"\n✓ Total BUSI images loaded: {len(kaggle_dataset)}")

    return kaggle_dataset


def load_and_merge_datasets():
    """Load and merge BUSI + HuggingFace datasets (your approach)"""
    print("\nLoading HuggingFace dataset...")

    # Load BUSI
    kaggle_dataset = load_busi_with_merged_masks()

    if kaggle_dataset is None:
        return None

    # Load HuggingFace dataset
    try:
        hf = load_dataset("Amss007/ultrasound_dataset_v3_1", cache_dir="../data")
        print(f"✓ HuggingFace dataset loaded: {hf}")

        features = Features(
            {
                "image": HfImage(),
                "conditioning_image": HfImage(),
                "caption": Value("string"),
            }
        )

        # Cast both datasets to same features
        kaggle_dataset = kaggle_dataset.cast(features)
        hf_dataset = hf["train"].cast(features)

        # Merge datasets
        merged_dataset = concatenate_datasets([kaggle_dataset, hf_dataset])

        print(f"✓ Merged dataset size: {len(merged_dataset)}")
        return merged_dataset

    except Exception as e:
        print(f"Warning: Could not load HF dataset: {e}")
        print("Continuing with BUSI only...")
        return kaggle_dataset


# ===============================================
# STEP 2: CONVERT MASKS TO BOUNDING BOXES
# ===============================================


class MaskToBBoxConverter:
    """Convert your merged masks to YOLO bounding boxes"""

    def __init__(self, output_dir="../data/yolo_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

    def setup_directories(self):
        """Create YOLO directory structure"""
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        print("✓ Created YOLO directory structure")

    def mask_to_yolo_bbox(self, mask_image):
        """
        Convert PIL mask image to YOLO bounding box format

        Args:
            mask_image: PIL Image (grayscale mask)

        Returns:
            [x_center, y_center, width, height] in normalized coordinates
        """
        if mask_image is None:
            return None

        try:
            # Convert PIL to numpy
            mask = np.array(mask_image)

            if mask.max() == 0:
                return None

            # Threshold to binary
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Get bounding box from largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Normalize coordinates
            img_h, img_w = mask.shape
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Validate
            if norm_w <= 0 or norm_h <= 0 or norm_w > 1 or norm_h > 1:
                return None

            return [x_center, y_center, norm_w, norm_h]

        except Exception as e:
            print(f"Warning: Error converting mask to bbox: {e}")
            return None

    def convert_dataset(self, dataset, train_ratio=0.7, val_ratio=0.15):
        """
        Convert your HuggingFace dataset to YOLO format

        Args:
            dataset: HuggingFace Dataset with your structure
        """
        print("\n" + "=" * 60)
        print("STEP 2: CONVERTING MASKS TO BOUNDING BOXES")
        print("=" * 60)

        if not SKLEARN_AVAILABLE:
            print("Error: scikit-learn required")
            return None

        self.setup_directories()

        # Split dataset
        indices = list(range(len(dataset)))
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio / (1 - train_ratio), random_state=42
        )

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        print(f"\nDataset split:")
        print(f"  Train: {len(train_idx)} images")
        print(f"  Val:   {len(val_idx)} images")
        print(f"  Test:  {len(test_idx)} images")

        stats = {"converted": 0, "with_boxes": 0, "without_boxes": 0, "skipped": 0}

        for split_name, split_indices in splits.items():
            print(f"\nProcessing {split_name} split...")

            for idx_num, dataset_idx in enumerate(split_indices):
                try:
                    # Get data from your dataset
                    example = dataset[dataset_idx]
                    image = example["image"]  # PIL Image
                    mask = example["conditioning_image"]  # PIL Image or None
                    caption = example["caption"]  # String

                    # Save image
                    img_filename = f"{split_name}_{idx_num:04d}.jpg"
                    img_path = self.images_dir / split_name / img_filename
                    image.save(img_path, quality=95)

                    # Convert mask to bounding box
                    bbox = self.mask_to_yolo_bbox(mask)

                    # Save YOLO label
                    label_filename = f"{split_name}_{idx_num:04d}.txt"
                    label_path = self.labels_dir / split_name / label_filename

                    with open(label_path, "w") as f:
                        if bbox and caption != "normal":
                            # class_id=0 for tumor, write bbox
                            f.write(
                                f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                            )
                            stats["with_boxes"] += 1
                        else:
                            # Empty file for normal images
                            stats["without_boxes"] += 1

                    stats["converted"] += 1

                    if (idx_num + 1) % 50 == 0:
                        print(
                            f"  Processed {idx_num + 1}/{len(split_indices)} images..."
                        )

                except Exception as e:
                    print(f"  Warning: Failed to process index {dataset_idx}: {e}")
                    stats["skipped"] += 1
                    continue

        print(f"\n✓ Conversion complete!")
        print(f"  Successfully converted: {stats['converted']}")
        print(f"  With bounding boxes:    {stats['with_boxes']}")
        print(f"  Without boxes (normal): {stats['without_boxes']}")
        print(f"  Skipped (errors):       {stats['skipped']}")

        # Create data.yaml
        self.create_yaml()

        return stats

    def create_yaml(self):
        """Create YOLO configuration file"""
        data_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["tumor"],
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"✓ Created {yaml_path}")


# ===============================================
# STEP 3-6: REUSE EXISTING TRAINING CODE
# ===============================================


class YOLOTrainer:
    """Train YOLO model"""

    def __init__(self, data_yaml="../data/yolo_dataset/data.yaml"):
        self.data_yaml = data_yaml
        self.model = None

    def load_model(self):
        """Load YOLOv8"""
        print("\n" + "=" * 60)
        print("STEP 3: LOADING YOLO MODEL")
        print("=" * 60)

        if not YOLO_AVAILABLE:
            print("Error: ultralytics not installed")
            return None

        try:
            self.model = YOLO("yolov8n.pt")
            print("✓ YOLOv8n loaded")
            return self.model
        except Exception as e:
            print(f"Error: {e}")
            return None

    def train(self, epochs=100, batch_size=16):
        """Train the model"""
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING YOLO")
        print("=" * 60)

        if not self.model:
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on: {device}")

        try:
            results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=batch_size,
                patience=20,
                device=device,
                augment=True,
                mosaic=0.3,
                mixup=0.0,
                hsv_h=0.01,
                hsv_s=0.3,
                hsv_v=0.2,
                degrees=10,
                flipud=0.5,
                fliplr=0.5,
                verbose=True,
            )

            print("\n✓ Training complete!")
            return results

        except Exception as e:
            print(f"Error: {e}")
            return None

    def validate(self):
        """Validate model"""
        print("\n" + "=" * 60)
        print("STEP 4: VALIDATION")
        print("=" * 60)

        if not self.model:
            return None

        try:
            metrics = self.model.val(data=self.data_yaml, split="test")

            print(f"\nMetrics:")
            print(f"  mAP50:     {metrics.box.map50:.4f}")
            print(f"  mAP50-95:  {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall:    {metrics.box.mr:.4f}")

            return metrics

        except Exception as e:
            print(f"Error: {e}")
            return None

    def visualize(self, num_samples=6):
        """Visualize predictions"""
        print("\nVisualizing predictions...")

        test_dir = Path(self.data_yaml).parent / "images" / "test"
        test_images = sorted(list(test_dir.glob("*.jpg")))[:num_samples]

        if not test_images:
            print("No test images found")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for ax, img_path in zip(axes, test_images):
            # Predict
            results = self.model.predict(str(img_path), conf=0.25, verbose=False)

            # Load and display
            img = PilImage.open(img_path)
            ax.imshow(img)

            # Draw boxes
            if results and len(results) > 0:
                for result in results:
                    if hasattr(result, "boxes") and result.boxes is not None:
                        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                            x1, y1, x2, y2 = box.cpu().numpy()
                            confidence = conf.cpu().item()

                            rect = patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                linewidth=2,
                                edgecolor="red",
                                facecolor="none",
                            )
                            ax.add_patch(rect)
                            ax.text(
                                x1,
                                y1 - 5,
                                f"{confidence:.2f}",
                                color="red",
                                fontsize=10,
                                bbox=dict(
                                    boxstyle="round", facecolor="white", alpha=0.7
                                ),
                            )

            ax.axis("off")
            ax.set_title(img_path.stem, fontsize=10)

        plt.tight_layout()
        plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
        print("✓ Saved to predictions.png")
        plt.show()


# ===============================================
# MAIN PIPELINE
# ===============================================


def main():
    """Complete pipeline using your dataset loading code"""

    print("=" * 60)
    print("TUMOR DETECTION PIPELINE")
    print("Using Your HuggingFace Dataset Approach")
    print("=" * 60)

    # STEP 1: Load data using your method
    dataset = load_and_merge_datasets()

    if dataset is None:
        print("Error: Could not load dataset")
        return

    # Visualize example (your code)
    print("\nVisualizing sample from dataset...")
    example = dataset[110]
    image = example["image"]
    mask = example["conditioning_image"]
    caption = example["caption"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis("off")

    if mask is not None:
        axs[1].imshow(mask)
        axs[1].set_title("Conditioning Image (Mask)")
    else:
        axs[1].set_visible(False)

    plt.suptitle(caption)
    plt.tight_layout()
    plt.savefig("dataset_sample.png", dpi=150, bbox_inches="tight")
    print("✓ Sample saved to dataset_sample.png")
    plt.show()

    # STEP 2: Convert masks to bounding boxes
    converter = MaskToBBoxConverter()
    stats = converter.convert_dataset(dataset)

    if not stats:
        return

    # STEP 3-4: Train and validate YOLO
    trainer = YOLOTrainer()
    model = trainer.load_model()

    if model:
        print("\n" + "=" * 60)
        print("TRAINING OPTIONS")
        print("=" * 60)
        print("Uncomment to train:")
        print("  trainer.train(epochs=100, batch_size=16)")
        print("=" * 60)

        # Uncomment to train:
        trainer.train(epochs=100, batch_size=16)

        # Validate (will be poor without training)
        trainer.validate()
        trainer.visualize(num_samples=6)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nYour approach integrated successfully:")
    print("  ✓ Used your dataset loading code")
    print("  ✓ Merged masks from multiple files")
    print("  ✓ Converted masks to YOLO bounding boxes")
    print("  ✓ Ready for YOLO training")


if __name__ == "__main__":
    main()
