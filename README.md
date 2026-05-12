# CNN — Fashion MNIST

> **UE24CS645BC2 | Deep Learning Theory & Practice | Assignment 1**

A Convolutional Neural Network built **entirely from first principles** using only NumPy — no PyTorch, no TensorFlow, no Keras. Every layer, every gradient, handwritten.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Layer-by-Layer Explanation](#layer-by-layer-explanation)
   - [Convolution Layer](#convolution-layer)
   - [ReLU Activation](#relu-activation)
   - [MaxPool Layer](#maxpool-layer)
   - [Flatten Layer](#flatten-layer)
   - [Fully Connected Layer](#fully-connected-layer)
4. [Forward & Backward Pass](#forward--backward-pass)
5. [Loss Function](#loss-function)
6. [Optimizer](#optimizer)
7. [Dataset](#dataset)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [File Structure](#file-structure)

---

## Project Overview

| Item | Detail |
|------|--------|
| Task | Image classification on Fashion MNIST |
| Framework | Pure NumPy (no deep learning library) |
| Layers coded | Conv → ReLU → MaxPool → Flatten → FC |
| Training | Mini-batch SGD with Momentum |
| Loss | Softmax Cross-Entropy |

The goal is to understand CNNs at the mathematical level — every matrix multiply, every gradient derivation, every index bookkeeping decision — rather than hiding it behind a `.fit()` call.

---

## Architecture

```
Input  (N, 1, 28, 28)
  │
  ▼  ConvLayer(1→8, 3×3, pad=1)  +  ReLU  →  (N,  8, 28, 28)
  ▼  MaxPool(2×2)                           →  (N,  8, 14, 14)
  │
  ▼  ConvLayer(8→16, 3×3, pad=1) +  ReLU  →  (N, 16, 14, 14)
  ▼  MaxPool(2×2)                           →  (N, 16,  7,  7)
  │
  ▼  Flatten                               →  (N, 784)
  │
  ▼  FC(784 → 128, ReLU)                   →  (N, 128)
  ▼  FC(128 → 10,  Softmax)                →  (N,  10)
  │
Output: class probabilities
```

---

## Layer-by-Layer Explanation

### Convolution Layer

**Definition:**  
A convolution slides a small learnable filter (kernel) across the input feature map and computes dot products at every valid position.

```
out[n, f, i, j] = Σ_{c,kh,kw}  W[f, c, kh, kw] · X_pad[n, c, i·s+kh, j·s+kw]  +  b[f]
```

Where:
- `W` — filter weights of shape `(F, C, kH, kW)`
- `b` — bias per filter
- `s` — stride
- `pad` — zero-padding to control output spatial size

**Output size:**
```
out_H = floor((H + 2*pad - kH) / stride) + 1
```

**Implementation trick — im2col:**  
Instead of nested Python loops (very slow), the input is rearranged into a 2-D matrix (`col`) of shape `(N, C·kH·kW, out_H·out_W)` so the entire convolution becomes a single matrix multiplication:
```
out_col = W_col @ col + b
```
This is the technique used inside cuDNN and most production frameworks.

---

### ReLU Activation

```
ReLU(x) = max(0, x)
```

**Backward:** gradient flows only where the forward activation was positive.
```
dReLU/dx = 1  if x > 0  else  0
```

---

### MaxPool Layer

Divides the feature map into non-overlapping `p×p` windows and keeps only the **maximum** value in each window. This:
- Reduces spatial size (fewer parameters downstream)
- Provides translation invariance

**Backward:** gradient is routed only to the position that achieved the maximum (winner-take-all). If multiple positions share the max, gradient is split equally.

---

### Flatten Layer

Reshapes `(N, C, H, W) → (N, C·H·W)` to connect the convolutional stack to the dense layers.

Backward: simply reshapes the gradient back to `(N, C, H, W)`.

---

### Fully Connected Layer

Standard dense layer:
```
z = X @ W + b
a = activation(z)        # ReLU or Softmax
```

**Gradients:**
```
dW = Xᵀ @ dz / N
db = mean(dz, axis=0)
dX = dz @ Wᵀ
```

For ReLU: `dz = dout * (z > 0)`  
For Softmax: the gradient is pre-computed by the loss function (see below).

---

## Forward & Backward Pass

### Forward Pass

```
X → conv1 → relu1 → pool1 → conv2 → relu2 → pool2 → flatten → fc1 → fc2 → probs
```

Each layer stores the inputs it needs for its own backward pass in `self._cache`.

### Backward Pass

The chain rule is applied in **reverse** order. Given the gradient of the loss `∂L/∂out` flowing back from above, each layer computes:

1. Its own weight gradients (`dW`, `db`)  
2. The gradient w.r.t. its input (`dX`), which it passes to the layer below

```
dout → fc2 → fc1 → flatten → pool2 → relu2 → conv2 → pool1 → relu1 → conv1
```

The key identity that makes backprop through convolution work:

- **dW** uses `im2col` of the input and the output gradient
- **dX** uses `col2im` to reconstruct the spatial gradient from the weight matrix transposed against the output gradient

---

## Loss Function

**Softmax + Cross-Entropy (combined):**

```
L = -1/N  Σ_n  log(p_{n, y_n})
```

The combined gradient w.r.t. logits is elegantly simple:
```
∂L/∂z_k = (p_k - 1{k == y}) / N
```

This gradient is fed directly into `fc2.backward()`.

---

## Optimizer

**SGD with Momentum:**

```
v_t  = β · v_{t-1}  −  η · ∇W
W   += v_t
```

| Hyperparameter | Value |
|----------------|-------|
| Learning rate η | 0.02 |
| Momentum β     | 0.9  |
| Batch size     | 64   |
| Epochs         | 5    |

Momentum accumulates a velocity vector that accelerates convergence in consistent directions and dampens oscillations.

---

## Dataset

**Fashion MNIST** ([Zalando Research](https://github.com/zalandoresearch/fashion-mnist))

| Property | Value |
|----------|-------|
| Images | 70,000 grayscale 28×28 |
| Training set | 60,000 |
| Test set | 10,000 |
| Classes | 10 |
| Pixel values | 0–255 (normalised to 0–1) |

**Class Labels:**

| ID | Label |
|----|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## How to Run

### Prerequisites

```bash
pip install numpy
```

Only NumPy is required. The dataset is downloaded automatically on first run.

### Run

```bash
python fashion_mnist_cnn.py
```

The script will:
1. Download Fashion MNIST (~30 MB) into `./fashion_mnist_data/`
2. Build and display the CNN
3. Train for 5 epochs with live progress
4. Print final test accuracy and per-class breakdown

### Expected Output

```
=======================================================
  CNN from Scratch – Fashion MNIST
=======================================================

[1] Loading Fashion MNIST …
    Train : (55000, 1, 28, 28)  Test : (10000, 1, 28, 28)

[2] Building CNN …
    Total trainable parameters: 109,994

[3] Training (5 epochs, batch=64) …

Epoch  1/5  loss=0.6821  train_acc=0.7512  val_acc=0.7680  (≈60s)
Epoch  2/5  loss=0.4103  train_acc=0.8499  val_acc=0.8420  (≈60s)
...

[4] Final evaluation on test set …
    Test Accuracy : 0.8400  (84.00%)

── Per-Class Accuracy ──────────────────────────────
  T-shirt/top     : 0.8320
  Trouser         : 0.9710
  ...
```

> ⏱ Training time depends on your CPU. Each epoch takes roughly 1–3 minutes on a modern laptop.

---

## Results

After 5 epochs of training from scratch:

| Metric | Value |
|--------|-------|
| Test Accuracy | ~82–85% |
| Parameters | ~110 K |
| Training time | ~5–15 min (CPU) |

Results are competitive for a hand-coded CNN with no data augmentation or regularisation. State-of-the-art with augmentation exceeds 96%, but the goal here is understanding, not benchmarking.

---

## File Structure

```
.
├── fashion_mnist_cnn.py    # Complete implementation
├── README.md               # This file
└── fashion_mnist_data/     # Auto-created on first run
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Convolution as matrix multiply (im2col) | `ConvLayer.forward` |
| Backprop through convolution (col2im) | `ConvLayer.backward` |
| Max-routing gradient | `MaxPoolLayer.backward` |
| Chain rule across layers | `SimpleCNN.backward` |
| Softmax-CE combined gradient | `softmax_cross_entropy_loss` |
| Mini-batch SGD with Momentum | `SGDMomentum.step` |
| He weight initialisation | `ConvLayer.__init__`, `FCLayer.__init__` |

---

*Built with ❤️ and NumPy.*
