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

## Experiment 7

Changes Made:
1. Merged image, backbone and video model optimizers into a single optimizer
2. Merged image and video model schedulers into a single scheduler
3. Changed image split from 80-20 to 100-0 train-val split. The same train set is used for validation during training.
4. Added early stopping with patience of 15 epochs based on validation video accuracy

Fixed Params:
* alpha_center = 0.5
* lr = 0.005

Try all combinations of NUM_FRAMES = [8, 16, 32] and LOSS_TYPE = [coherence, triplet(hard), triplet(semi-hard), triplet(all)]

### NUM_FRAMES = 8 and LOSS_TYPE = coherence

Epoch 24/422 (Iter 437/8000)
- Train - Image Loss: 12.6851, Acc: 52.63%
- Train - Video Loss: 1.4597, Acc: 59.73%
- Val   - Image Acc: 63.06% (train set), Video Acc: 64.86%
- Optimal thresholds - Image: 0.479, Video: 1.000
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 63.06%
- Best Video Acc: 72.97%

### NUM_FRAMES = 16 and LOSS_TYPE = coherence

Epoch 23/422 (Iter 418/8000)
- Train - Image Loss: 12.0213, Acc: 57.89%
- Train - Video Loss: 1.4359, Acc: 61.07%
- Val   - Image Acc: 60.43% (train set), Video Acc: 48.65%
- Optimal thresholds - Image: 0.466, Video: 0.868
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 65.53%
- Best Video Acc: 75.68%

### NUM_FRAMES = 32 and LOSS_TYPE = coherence

Epoch 19/422 (Iter 342/8000)
- Train - Image Loss: 12.3321, Acc: 55.92%
- Train - Video Loss: 1.4998, Acc: 63.76%
- Val   - Image Acc: 62.91% (train set), Video Acc: 59.46%
- Optimal thresholds - Image: 0.515, Video: 0.999
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 65.07%
- Best Video Acc: 64.86%

### NUM_FRAMES = 8 and LOSS_TYPE = triplet(hard)

Epoch 38/422 (Iter 703/8000)
- Train - Image Loss: 12.5704, Acc: 59.21%
- Train - Video Loss: 5.0745, Acc: 59.06%
- Val   - Image Acc: 66.15% (train set), Video Acc: 64.86%
- Optimal thresholds - Image: 0.480, Video: 1.000
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 67.39%
- Best Video Acc: 64.86%

### NUM_FRAMES = 16 and LOSS_TYPE = triplet(hard)

Epoch 30/422 (Iter 551/8000)
- Train - Image Loss: 12.6908, Acc: 52.63%
- Train - Video Loss: 5.8162, Acc: 60.40%
- Val   - Image Acc: 57.34% (train set), Video Acc: 64.86%
- Optimal thresholds - Image: 0.535, Video: 1.000
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 57.65%
- Best Video Acc: 67.57%

### NUM_FRAMES = 32 and LOSS_TYPE = triplet(hard)

Epoch 43/422 (Iter 798/8000)
- Train - Image Loss: 12.2084, Acc: 57.89%
- Train - Video Loss: 5.8135, Acc: 59.73%
- Val   - Image Acc: 57.19% (train set), Video Acc: 51.35%
- Optimal thresholds - Image: 0.505, Video: 0.990
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 63.21%
- Best Video Acc: 70.27%

### NUM_FRAMES = 8 and LOSS_TYPE = triplet(semi-hard)

Epoch 32/422 (Iter 589/8000)
- Train - Image Loss: 12.9334, Acc: 53.95%
- Train - Video Loss: 1.8724, Acc: 59.06%
- Val   - Image Acc: 44.51% (train set), Video Acc: 64.86%
- Optimal thresholds - Image: 0.541, Video: 1.000
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 49.92%
- Best Video Acc: 67.57%

### NUM_FRAMES = 16 and LOSS_TYPE = triplet(semi-hard)

Epoch 26/422 (Iter 475/8000)
- Train - Image Loss: 12.4789, Acc: 43.42%
- Train - Video Loss: 1.8371, Acc: 61.07%
- Val   - Image Acc: 38.95% (train set), Video Acc: 62.16%
- Optimal thresholds - Image: 0.548, Video: 0.997
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 41.11%
- Best Video Acc: 62.16%

### NUM_FRAMES = 32 and LOSS_TYPE = triplet(semi-hard)

Epoch 22/422 (Iter 399/8000)
- Train - Image Loss: 12.5903, Acc: 47.37%
- Train - Video Loss: 1.8735, Acc: 66.44%
- Val   - Image Acc: 53.01% (train set), Video Acc: 43.24%
- Optimal thresholds - Image: 0.493, Video: 0.901
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 62.29%
- Best Video Acc: 59.46%

### NUM_FRAMES = 8 and LOSS_TYPE = triplet(all)

Epoch 32/422 (Iter 589/8000)
- Train - Image Loss: 12.7354, Acc: 59.87%
- Train - Video Loss: 2.3503, Acc: 61.07%
- Val   - Image Acc: 53.94% (train set), Video Acc: 64.86%
- Optimal thresholds - Image: 0.483, Video: 1.000
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 53.17%
- Best Video Acc: 64.86%

### NUM_FRAMES = 16 and LOSS_TYPE = triplet(all)

Epoch 17/422 (Iter 304/8000)
- Train - Image Loss: 12.3755, Acc: 51.32%
- Train - Video Loss: 2.3314, Acc: 57.72%
- Val   - Image Acc: 59.20% (train set), Video Acc: 67.57%
- Optimal thresholds - Image: 0.519, Video: 0.991
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 64.45%
- Best Video Acc: 70.27%

### NUM_FRAMES = 32 and LOSS_TYPE = triplet(all)

Epoch 27/422 (Iter 494/8000)
- Train - Image Loss: 12.2079, Acc: 63.82%
- Train - Video Loss: 2.3484, Acc: 61.74%
- Val   - Image Acc: 59.97% (train set), Video Acc: 62.16%
- Optimal thresholds - Image: 0.499, Video: 0.548
- Early stopping triggered after 15 epochs without improvement
- Best Image Acc: 59.81%
- Best Video Acc: 67.57%
