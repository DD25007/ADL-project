# ResNet-50 Architecture Overview

```
Input Image (224×224×3)
↓
Initial Block:
- Conv1: 7×7, 64 filters, stride=2, padding=3
- BatchNorm + ReLU
- MaxPool: 3×3, stride=2, padding=1
Output: 56×56×64

Stage 1 (×3 blocks):
[Block 1] - Projection shortcut
- 1×1, 64
- 3×3, 64
- 1×1, 256
- Shortcut: 1×1 conv, 256 (to match dimensions)
[Block 2-3] - Identity shortcuts
- Same convolution pattern as Block 1
Output: 56×56×256

Stage 2 (×4 blocks):
[Block 1] - Projection shortcut
- 1×1, 128, stride=2 (downsampling)
- 3×3, 128
- 1×1, 512
- Shortcut: 1×1 conv, 512, stride=2
[Block 2-4] - Identity shortcuts
- Same pattern but stride=1
Output: 28×28×512

Stage 3 (×6 blocks):
[Block 1] - Projection shortcut
- 1×1, 256, stride=2 (downsampling)
- 3×3, 256
- 1×1, 1024
- Shortcut: 1×1 conv, 1024, stride=2
[Block 2-6] - Identity shortcuts
- Same pattern but stride=1
Output: 14×14×1024

Stage 4 (×3 blocks):
[Block 1] - Projection shortcut
- 1×1, 512, stride=2 (downsampling)
- 3×3, 512
- 1×1, 2048
- Shortcut: 1×1 conv, 2048, stride=2
[Block 2-3] - Identity shortcuts
- Same pattern but stride=1
Output: 7×7×2048

Final Layers:
- Global Average Pooling
Output: 1×1×2048
- Fully Connected (1000)
Output: 1000 (ImageNet classes)
```

## Residual Block Structure

### Bottleneck Block Pattern:
```
Input
├─> 1×1 conv (reduce dimensions)
│   └─> BatchNorm + ReLU
├─> 3×3 conv
│   └─> BatchNorm + ReLU
├─> 1×1 conv (expand dimensions)
│   └─> BatchNorm
│
└─> Skip Connection (identity or projection)
    │
    └─> Element-wise Addition
        └─> ReLU
Output
```

### Two Types of Skip Connections:
1. **Identity Shortcuts**: When input/output dimensions match (most blocks)
2. **Projection Shortcuts**: When dimensions change (first block of each stage)
   - Uses 1×1 conv to match channel dimensions
   - Applies stride=2 when spatial downsampling occurs

## Architecture Statistics

### Total Layers: 50
- 1 initial 7×7 conv layer
- 48 conv layers in 16 residual blocks (3 convs per block)
- 1 fully connected layer

### Parameters per Stage:
1. **Stage 1**: ~0.2M parameters
2. **Stage 2**: ~1.2M parameters
3. **Stage 3**: ~7.1M parameters
4. **Stage 4**: ~14.9M parameters
5. **Other layers**: ~2.2M parameters

**Total Parameters: ~25.6M**

### Feature Map Size Progression:
```
224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7 → 1×1
    ↓         ↓        ↓        ↓        ↓       ↓
  Conv1   MaxPool  Stage1  Stage2   Stage3  Stage4  GAP
```

## Key Design Principles

### 1. **Downsampling Strategy**
- Initial reduction: 7×7 conv (stride=2) + 3×3 maxpool (stride=2)
- Subsequent stages: First block uses stride=2 in both 1×1 and 3×3 convs

### 2. **Skip Connections**
- Enable gradient flow through deep networks
- Mitigate vanishing gradient problem
- Allow learning of residual functions F(x) instead of H(x)

### 3. **Bottleneck Design**
- 1×1 conv reduces dimensions (computational efficiency)
- 3×3 conv operates on reduced dimensions
- 1×1 conv expands back to required dimensions
- Reduces parameters while maintaining representational power

### 4. **Batch Normalization**
- Applied after every convolutional layer
- Before activation function (except after final addition)
- Stabilizes training and enables higher learning rates

### 5. **Activation Function**
- ReLU used throughout
- Applied after BatchNorm
- Final ReLU after skip connection addition

## Computational Efficiency

- **Bottleneck design** reduces computational cost by ~4x compared to basic residual blocks
- **Parameter sharing** through 1×1 convolutions
- **Strategic downsampling** maintains reasonable feature map sizes
- **Global Average Pooling** eliminates need for large fully connected layers

## Performance Characteristics

- **ImageNet Top-5 Error**: ~7.1%
- **ImageNet Top-1 Error**: ~23.9%
- **Computational Complexity**: ~3.8 GFLOPs
- **Training**: Converges efficiently despite depth due to skip connections
