# Visual & Text Prompt-Based Video Clip Detection (IIT Mandi Advanced Deep Learning Project)

## Project Overview

Foundation model-based framework for detecting and classifying objects in video clips using visual and text prompts with DINO-Family models.

**Course:** CS672 - Advanced Deep Learning  
**Project ID:** P1 (Sub-problem)  
**Mentor:** Richa

## Problem Statement

Traditional object detection models struggle with false detections and mislocalization when dealing with similar structures in specialized domains. Current approaches often fail on limited or imbalanced datasets and require full retraining for new tasks. Our system addresses these challenges by implementing visual and text prompting to guide pre-trained foundation models toward accurate detection in video sequences without full model retraining.

## Key Features

- **Multi-Modal Prompting**: Combines visual prompts (example regions/images) and text prompts (semantic descriptions)
- **Video Processing**: Processes video inputs to extract meaningful clips with detected objects
- **Foundation Model Integration**: Built on DINO-Family architectures for robust feature extraction
- **Flexible Deployment**: Works with various natural image datasets and ultrasound imagery

## Methodology

**Model Architecture**: DINO-Family (DINOv2, Grounding-DINO, DINO-X) with dual-mode prompt integration

**Training Strategy**:
- Train on annotated images with bounding boxes
- Apply temporal processing for video clip extraction
- Implement prompt tuning for domain alignment

**Image-to-Video Justification**: Spatial features learned from images transfer effectively to video frames. This approach leverages abundant annotated image datasets while maintaining computational efficiency compared to direct video training.

## Dataset

* [Kaggle Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
* [Hugging Face](https://huggingface.co/datasets/Amss007/ultrasound_dataset_v3_1)

Due to licensing constraints with the original gall bladder cancer dataset:
- **Alternative Sources**: Kaggle publicly available ultrasound images
- **Format**: Bounding box annotations for supervised learning

## Evaluation Metrics

- **Detection**: Precision, Recall, F1-score for both YOLO and DINO components
- **Video-Specific**: Temporal consistency, clip extraction accuracy
- **Ablation**: YOLO-only vs DINO-only vs Hybrid performance comparison
- **Prompt Effectiveness**: Performance across prompt types (text vs visual vs combined)

## Project Structure

```
├── src/models/           # DINO-family implementations
├── src/prompting/        # Prompt handlers
├── src/video_processing/ # Video-to-clip extraction
├── configs/              # Model configurations
├── data/                 # Dataset utilities
└── notebooks/            # Experiments
```

## References

- [DINO-X API](https://github.com/IDEA-Research/DINO-X-API)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Code and Implementation of Grounding DINO](https://github.com/open-mmlab/mmdetection)

### Contributors
* Kaustav Goswami (DD25007)
* Mitanshi (S25046)
* Lucky Rathore (S25021)
