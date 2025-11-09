# Parameter Tuning Obeservations

## FIXED Params (to match the paper)

* TOTAL_ITERATIONS = 8000
* IMAGE_BATCH_SIZE = 8
* VIDEO_BATCH_SIZE = 8
* LEARNING_RATE = 0.005


## Experiment 1

For
> * Loss = coherence
> * alpha center = 0.5
> * NUM_FRAMES = 32

### alpha = 0.2 (Overfitting for Video)

* Train - Image Loss: 0.0785, Acc: 100.00%
* Train - Video Loss: 0.0044, Acc: 100.00%
* Val   - Image Acc: 83.08%, Video Acc: 54.05%
* **Best Image Acc**: 90.77%
* **Best Video Acc**: 64.86%

### alpha = 0.5 (Overfitting for Video)

* Train - Image Loss: 0.0983, Acc: 100.00%
* Train - Video Loss: 0.0085, Acc: 100.00%
* Val   - Image Acc: 85.38%, Video Acc: 56.76%
* **Best Image Acc**: 86.15%
* **Best Video Acc**: 67.57%

### alpha = 0.8 (Overfitting for Video)

* Train - Image Loss: 0.2342, Acc: 87.50%
* Train - Video Loss: 0.0076, Acc: 100.00%
* Val   - Image Acc: 83.85%, Video Acc: 51.35%
* **Best Image Acc**: 88.46%
* **Best Video Acc**: 70.27%

## Experiment 2

For
> * Loss = coherence
> * alpha center = 0.5
> * NUM_FRAMES = 32

Try all combinations of alpha = [0.2, 0.5, 0.8] and NUM_FRAMES = [8, 16, 32]

### alpha = 0.2 and NUM_FRAMES = 8 (Overfitting for Video)

* Train - Image Loss: 0.0789, Acc: 100.00%
* Train - Video Loss: 0.0048, Acc: 100.00%
* Val   - Image Acc: 84.62%, Video Acc: 45.95%
* **Best Image Acc**: 89.23%
* **Best Video Acc**: 67.57%


### alpha = 0.2 and NUM_FRAMES = 16 (Overfitting for Video)

* Train - Image Loss: 0.1064, Acc: 100.00%
* Train - Video Loss: 0.0360, Acc: 100.00%
* Val   - Image Acc: 91.54%, Video Acc: 48.65%
* **Best Image Acc**: 93.08%
* **Best Video Acc**: 64.86%

> Stopping here as the other combinations will also overfit for video and do not improve accuracy

## Experiment 3

Changes Made:

1. Enhanced Data Augmentation
   ```python
   # Enhance the existing train transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Add rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Increase jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
        transforms.RandomErasing(p=0.2),  # Add random erasing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   ```

2. Increased Dropout from 0.3
    ```python
    # In ImageClassificationNetwork
    self.feature_layer = nn.Sequential(
        nn.Linear(2048, feature_dim),
        nn.ReLU(),
        nn.Dropout(0.5)  # Increase from 0.3
    )

    # In KGANet classifier
    self.classifier = nn.Sequential(
        nn.Dropout(0.6),  # Increase from 0.5
        nn.Linear(feature_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.4),  # Increase from 0.3
        nn.Linear(512, num_classes)
    )
    ```

3. Increased weight decay in image and video head optimizer to 1e-3 from 1e-4
    ```python
    image_head_optimizer = torch.optim.SGD(
        image_head_params,
        lr=image_lr,
        momentum=0.9,
        weight_decay=1e-3  # Increase from 1e-4
    )

    video_head_optimizer = torch.optim.SGD(
        video_head_params,
        lr=video_lr,
        momentum=0.9,
        weight_decay=1e-3  # Increase from 1e-4
    )
    ```

For
> * Loss = coherence
> * alpha center = 0.5
> * NUM_FRAMES = 32

Try all combinations of alpha = [0.2, 0.5, 0.8] and NUM_FRAMES = [8, 16, 32]

### alpha = 0.2 and NUM_FRAMES = 8

* Train - Image Loss: 0.2147, Acc: 100.00%
* Train - Video Loss: 0.0107, Acc: 100.00%
* Val   - Image Acc: 88.46%, Video Acc: 43.24%
* **Best Image Acc**: 93.08%
* **Best Video Acc**: 64.86%


### alpha = 0.2 and NUM_FRAMES = 16

* Train - Image Loss: 0.1936, Acc: 100.00%
* Train - Video Loss: 0.0131, Acc: 100.00%
* Val   - Image Acc: 87.69%, Video Acc: 43.24%
* **Best Image Acc**: 90.77%
* **Best Video Acc**: 67.57%

### alpha = 0.2 and NUM_FRAMES = 32

* Train - Image Loss: 0.2152, Acc: 100.00%
* Train - Video Loss: 0.0110, Acc: 100.00%
* Val   - Image Acc: 87.69%, Video Acc: 48.65%
* **Best Image Acc**: 93.08%
* **Best Video Acc**: 64.86%

### alpha = 0.5 and NUM_FRAMES = 8

* Train - Image Loss: 0.1199, Acc: 100.00%
* Train - Video Loss: 0.0259, Acc: 100.00%
* Val   - Image Acc: 87.69%, Video Acc: 54.05%
* **Best Image Acc**: 90.77%
* **Best Video Acc**: 70.27%


### alpha = 0.5 and NUM_FRAMES = 16

* Train - Image Loss: 0.1275, Acc: 100.00%
* Train - Video Loss: 0.0148, Acc: 100.00%
* Val   - Image Acc: 79.23%, Video Acc: 54.05%
* **Best Image Acc**: 88.46%
* **Best Video Acc**: 67.57%

### alpha = 0.5 and NUM_FRAMES = 32

* Train - Image Loss: 0.2095, Acc: 100.00%
* Train - Video Loss: 0.0102, Acc: 100.00%
* Image Acc: 87.69%, Video Acc: 54.05%
* **Best Image Acc**: 93.08%
* **Best Video Acc**: 67.57%

### alpha = 0.8 and NUM_FRAMES = 8

* Train - Image Loss: 0.1751, Acc: 100.00%
* Train - Video Loss: 0.0096, Acc: 100.00%
* Val   - Image Acc: 87.69%, Video Acc: 54.05%
* **Best Image Acc**: 90.77%
* **Best Video Acc**: 64.86%


### alpha = 0.8 and NUM_FRAMES = 16

* Train - Image Loss: 0.1282, Acc: 100.00%
* Train - Video Loss: 0.0164, Acc: 100.00%
* Val   - Image Acc: 81.54%, Video Acc: 45.95%
* **Best Image Acc**: 89.23%
* **Best Video Acc**: 70.27%

### alpha = 0.8 and NUM_FRAMES = 32

* Train - Image Loss: 0.1378, Acc: 100.00%
* Train - Video Loss: 0.0181, Acc: 100.00%
* Val   - Image Acc: 83.08%, Video Acc: 56.76%
* **Best Image Acc**: 90.77%
* **Best Video Acc**: 70.27%


## Experiment 4

Changes Made:

1. In ImageClassificationNetwork, added normalization layer
from:
```python
self.feature_layer = nn.Sequential(
    nn.Linear(2048, feature_dim),
    nn.ReLU(),
    nn.Dropout(0.5),  # Increase from 0.3
)
```
to:
```python
self.feature_layer = nn.Sequential(
    nn.Linear(2048, feature_dim),
    nn.LayerNorm(feature_dim),  # Add normalization
    nn.ReLU(),
    nn.Dropout(0.5),
)
```

For
> * Loss = coherence
> * alpha center = 0.5
> * NUM_FRAMES = 32

Try all combinations of alpha = [0.2, 0.5, 0.8] and NUM_FRAMES = [8, 16, 32]

### alpha = 0.2 and NUM_FRAMES = 8

* Train - Image Loss: 0.2126, Acc: 100.00%
* Train - Video Loss: 0.0155, Acc: 100.00%
* Val   - Image Acc: 87.69%, Video Acc: 51.35%
* **Best Image Acc**: 92.31%
* **Best Video Acc**: 70.27%


### alpha = 0.2 and NUM_FRAMES = 16

* Train - Image Loss: 0.2878, Acc: 97.37%
* Train - Video Loss: 0.1058, Acc: 96.64%
* Val   - Image Acc: 87.69%, Video Acc: 62.16%
* **Best Image Acc**: 93.08%
* **Best Video Acc**: 75.68%

### alpha = 0.2 and NUM_FRAMES = 32

* Train - Image Loss: 0.2132, Acc: 100.00%
* Train - Video Loss: 0.0146, Acc: 100.00%
* Val   - Image Acc: 91.54%, Video Acc: 56.76%
* **Best Image Acc**: 93.85%
* **Best Video Acc**: 70.27%


## Experiment 5
Changes Made:
* Fixed cohrence loss implementation
* Added frame attention weights in KGA-net

Try all combinations of alpha = [0.2, 0.5, 0.8] and NUM_FRAMES = [8, 16, 32]

### alpha = 0.2 and NUM_FRAMES = 8

* Train - Image Loss: 0.5851, Acc: 62.50%
* Train - Video Loss: 1.4161, Acc: 62.50%
* Val   - Image Acc: 74.62%, Video Acc: 54.05%
* **Best Image Acc**: 74.62%
* **Best Video Acc**: 75.68%

### alpha = 0.2 and NUM_FRAMES = 16

* Train - Image Loss: 0.3913, Acc: 87.50%
* Train - Video Loss: 1.3998, Acc: 62.50%
* Val   - Image Acc: 86.15%, Video Acc: 72.97%
* **Best Image Acc**: 88.46%
* **Best Video Acc**: 75.68%

### alpha = 0.5 and NUM_FRAMES = 8

* Train - Image Loss: 0.6655, Acc: 62.50%
* Train - Video Loss: 1.4883, Acc: 50.00%
* Val   - Image Acc: 72.31%, Video Acc: 62.16%
* **Best Image Acc**: 73.08%
* **Best Video Acc**: 78.38%

### alpha = 0.5 and NUM_FRAMES = 16

* Train - Image Loss: 0.6654, Acc: 62.50%
* Train - Video Loss: 1.4895, Acc: 37.50%
* Val   - Image Acc: 72.31%, Video Acc: 70.27%
* **Best Image Acc**: 72.31%
* **Best Video Acc**: 78.38%

### alpha = 0.8 and NUM_FRAMES = 8

* Train - Image Loss: 0.6651, Acc: 62.50%
* Train - Video Loss: 1.5221, Acc: 50.00%
* Val   - Image Acc: 72.31%, Video Acc: 56.76%
* **Best Image Acc**: 72.31%
* **Best Video Acc**: 72.97%

### alpha = 0.8 and NUM_FRAMES = 16

* Train - Image Loss: 0.6639, Acc: 62.50%
* Train - Video Loss: 1.5905, Acc: 37.50%
* Val   - Image Acc: 72.31%, Video Acc: 59.46%
* **Best Image Acc**: 72.31%
* **Best Video Acc**: 72.97%



## Experiment 6

Changes Made:
1. Changed Image model to 2048 from 512 dimension
2. Added Youden index based optimal thresholding after each validation epoch 

Fixed Params:
* alhpa_video = 1
* lr = 0.005

Try all combinations of alpha_center = [0.1, 0.5, 1] and NUM_FRAMES = [8, 16]

### alpha_center = 0.1 and NUM_FRAMES = 8
Last Epoch Results:
- Train - Image Loss: 0.0221, Acc: 100.00%
- Train - Video Loss: 0.0211, Acc: 100.00%
- Val   - Image Acc: 73.85%, Video Acc: 54.05%
- Optimal thresholds - Image: 0.988, Video: 0.000
- Best Image Acc: 83.85%
- Best Video Acc: 70.27%

### alpha_center = 0.1 and NUM_FRAMES = 16

Last Epoch Results:
- Train - Image Loss: 0.0225, Acc: 100.00%
- Train - Video Loss: 0.0241, Acc: 100.00%
- Val   - Image Acc: 81.54%, Video Acc: 54.05%
- Optimal thresholds - Image: 0.863, Video: 0.000
- Best Image Acc: 89.23%
- Best Video Acc: 72.97%

### alpha_center = 0.5 and NUM_FRAMES = 8

Last Epoch Results:
- Train - Image Loss: 0.0707, Acc: 100.00%
- Train - Video Loss: 1.5196, Acc: 37.50%
- Val   - Image Acc: 27.69%, Video Acc: 64.86%
- Optimal thresholds - Image: 0.869, Video: 0.589
- Best Image Acc: 83.85%
- Best Video Acc: 67.57%

### alpha_center = 0.5 and NUM_FRAMES = 16

Last Epoch Results:
- Train - Image Loss: 0.0571, Acc: 100.00%
- INFO - Train - Video Loss: 0.0215, Acc: 100.00%
- INFO - Val   - Image Acc: 89.23%, Video Acc: 51.35%
- INFO - Optimal thresholds - Image: 0.491, Video: 1.000
- INFO - Best Image Acc: 92.31%
- INFO - Best Video Acc: 70.27%

### alpha_center = 1 and NUM_FRAMES = 8

Last Epoch Results:
- Train - Image Loss: 0.1377, Acc: 100.00%
- Train - Video Loss: 0.0225, Acc: 100.00%
- Val   - Image Acc: 90.00%, Video Acc: 56.76%
- Optimal thresholds - Image: 0.454, Video: 1.000
- Best Image Acc: 93.08%
- Best Video Acc: 70.27%

### alpha_center = 1 and NUM_FRAMES = 16

Last Epoch Results:
- Train - Image Loss: 0.6677, Acc: 62.50%
- Train - Video Loss: 1.5292, Acc: 37.50%
- Val   - Image Acc: 72.31%, Video Acc: 64.86%
- Optimal thresholds - Image: 0.336, Video: 0.584
- Best Image Acc: 74.62%
- Best Video Acc: 67.57%