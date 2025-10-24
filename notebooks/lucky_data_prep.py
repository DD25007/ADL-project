# lucky_data_prep.py

import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PilImage
from sklearn.model_selection import train_test_split
from datasets import (
    Dataset, Features, Value, Image as HfImage, load_dataset, concatenate_datasets
)
from io import BytesIO

# --- STEP 1: LOAD DATA ---

def load_busi_with_merged_masks(base_dir="../data/Dataset_BUSI_with_GT/"):
    """
    Loads BUSI dataset, merges masks, and creates a HuggingFace Dataset.
    Stores image and mask data as file PATHS for robust processing.
    """
    print("\n" + "=" * 60)
    print("STEP 1: LOADING BUSI DATASET")
    print(f"Checking for RAW data at: {os.path.abspath(base_dir)}")
    print("=" * 60)
    if not os.path.exists(base_dir):
        print(f"✗ FATAL ERROR: Raw Dataset folder not found at {base_dir}")
        return None

    records = []
    for label, class_id in [("benign", 0), ("malignant", 1), ("normal", 2)]:
        folder = os.path.join(base_dir, label)
        if not os.path.exists(folder):
            continue
        
        yolo_class_id = 0 

        for file in os.listdir(folder):
            if not file.endswith(".png") or "mask" in file:
                continue

            img_path = os.path.join(folder, file)
            base_name = file.replace(".png", "")
            
            # Look for mask files
            mask_files = [
                os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(base_name + "_mask")
            ]
            
            mask_path = mask_files[0] if mask_files else None

            records.append(
                {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "caption": label,
                    "yolo_class_id": yolo_class_id,
                    "has_tumor": bool(mask_files) 
                }
            )

    features = Features(
        {
            "image_path": Value("string"),
            "mask_path": Value("string"),
            "caption": Value("string"),
            "yolo_class_id": Value("int32"),
            "has_tumor": Value("bool")
        }
    )

    kaggle_dataset = Dataset.from_list(records, features=features)
    print(f"\nTotal BUSI images loaded: {len(kaggle_dataset)}")
    return kaggle_dataset


def load_huggingface_dataset():
    """Load HuggingFace breast ultrasound dataset and convert to BUSI format."""
    print("\nLoading HuggingFace dataset (nielsr/breast-cancer)...")
    try:
        hf_dataset = load_dataset("nielsr/breast-cancer", split="train")
        print(f"Loaded {len(hf_dataset)} images from HuggingFace")
        
        # Convert HF dataset to match BUSI format
        converted_records = []
        temp_dir = Path("../data/temp_hf_images")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, item in enumerate(hf_dataset):
            try:
                # Save image temporarily
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_path = temp_dir / f"hf_{idx:04d}.png"
                img.save(img_path)
                
                # HF dataset has binary labels: 0=benign, 1=malignant
                label = "malignant" if item.get('labels', 0) == 1 else "benign"
                
                converted_records.append({
                    "image_path": str(img_path),
                    "mask_path": None,  # HF dataset doesn't have masks
                    "caption": label,
                    "yolo_class_id": 0,
                    "has_tumor": True  # Assume all HF images have tumors
                })
                
            except Exception as e:
                print(f"  Warning: Skipped HF image {idx}: {e}")
                continue
        
        if not converted_records:
            print("No valid records from HuggingFace dataset")
            return None
            
        features = Features({
            "image_path": Value("string"),
            "mask_path": Value("string"),
            "caption": Value("string"),
            "yolo_class_id": Value("int32"),
            "has_tumor": Value("bool")
        })
        
        hf_converted = Dataset.from_list(converted_records, features=features)
        print(f"Converted {len(hf_converted)} HuggingFace images")
        return hf_converted
        
    except Exception as e:
        print(f"Could not load HuggingFace dataset: {e}")
        return None


def load_and_merge_datasets():
    """Load and merge BUSI + HuggingFace datasets."""
    # Load BUSI dataset
    busi_dataset = load_busi_with_merged_masks()
    if busi_dataset is None:
        return None

    # Load HuggingFace dataset
    hf_dataset = load_huggingface_dataset()
    
    if hf_dataset is None:
        print("\nWarning: Using only BUSI dataset (HF merge failed)")
        return busi_dataset
    
    # Merge datasets
    try:
        merged_dataset = concatenate_datasets([busi_dataset, hf_dataset])
        print(f"\nSuccessfully merged datasets!")
        print(f"  - BUSI: {len(busi_dataset)} images")
        print(f"  - HuggingFace: {len(hf_dataset)} images")
        print(f"  - Total: {len(merged_dataset)} images")
        return merged_dataset
    except Exception as e:
        print(f"\nMerge failed: {e}")
        print("  Using only BUSI dataset")
        return busi_dataset


# --- STEP 2: CONVERT MASKS TO BOUNDING BOXES (YOLO Format) ---

class MaskToBBoxConverter:
    """Converts masks to YOLO bounding boxes and creates the dataset structure."""
    def __init__(self, output_dir="../data/yolo_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    def setup_directories(self):
        """Create YOLO directory structure"""
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        print("Created YOLO directory structure")

    def mask_to_yolo_bbox(self, mask_path, img_path):
        """Convert PIL mask image to YOLO bounding box format"""
        if mask_path is None or mask_path == "None":
            return None
        
        try:
            mask_img = PilImage.open(mask_path).convert("L")
            mask = np.array(mask_img)
            
            img = PilImage.open(img_path).convert("RGB")
            img_w, img_h = img.size 
            
        except Exception:
            return None

        if mask.ndim < 2 or mask.max() == 0:
            return None

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if img_w == 0 or img_h == 0:
            return None
            
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        return [x_center, y_center, norm_w, norm_h]
    
    def create_synthetic_bbox(self, img_path):
        """Create a centered bbox for images without masks (HF dataset)"""
        try:
            img = PilImage.open(img_path).convert("RGB")
            img_w, img_h = img.size
            
            # Create a bbox covering central 60% of image
            x_center = 0.5
            y_center = 0.5
            width = 0.6
            height = 0.6
            
            return [x_center, y_center, width, height]
        except Exception:
            return None
        
    def create_yaml(self):
        """Create YOLO configuration file"""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['tumor']
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        print(f"Created data.yaml at {yaml_path}")
        
    def convert_dataset(self, dataset):
        """Process dataset and save files in YOLO format."""
        print("\n" + "=" * 60)
        print("STEP 2: CONVERTING TO YOLO FORMAT")
        print("=" * 60)
        self.setup_directories()
        
        records = dataset.to_list()
        
        train_records, temp_records = train_test_split(records, train_size=self.split_ratios["train"], random_state=42)
        val_records, test_records = train_test_split(temp_records, 
                                                     train_size=self.split_ratios["val"]/(1-self.split_ratios["train"]), 
                                                     random_state=42)

        splits = {'train': train_records, 'val': val_records, 'test': test_records}
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        print(f"Total Records: {len(records)}")
        print(f"Split sizes: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

        for split_name, split_records in splits.items():
            print(f"\nProcessing {split_name} split...")
            for idx, record in enumerate(split_records):
                image_path_str = record['image_path']
                mask_path_str = record['mask_path']
                
                original_filename = Path(image_path_str).name

                try:
                    img = PilImage.open(image_path_str).convert('RGB')
                    img_filename = f"{split_name}_{idx:04d}_{original_filename.replace('.png', '.jpg')}"
                    img_path = self.images_dir / split_name / img_filename
                    
                    (self.images_dir / split_name).mkdir(parents=True, exist_ok=True)
                    (self.labels_dir / split_name).mkdir(parents=True, exist_ok=True)
                    
                    img.save(img_path)

                    label_filename = f"{split_name}_{idx:04d}_{original_filename.replace('.png', '.txt')}"
                    label_path = self.labels_dir / split_name / label_filename
                    
                    # Try to get bbox from mask, or create synthetic one
                    if mask_path_str and mask_path_str != "None":
                        bbox_normalized = self.mask_to_yolo_bbox(mask_path_str, image_path_str)
                    else:
                        # For HF images without masks, create synthetic bbox
                        bbox_normalized = self.create_synthetic_bbox(image_path_str)
                    
                    with open(label_path, 'w') as f:
                        if record['has_tumor'] and bbox_normalized:
                            f.write(f"0 {bbox_normalized[0]:.6f} {bbox_normalized[1]:.6f} {bbox_normalized[2]:.6f} {bbox_normalized[3]:.6f}\n")

                    stats[split_name] += 1

                except Exception as e:
                    print(f"Failed on {split_name} image {original_filename}: {e}")
                    continue 

        print(f"\nConversion complete:")
        for split, count in stats.items():
            print(f"  {split}: {count} images")
            
        self.create_yaml()
        return stats