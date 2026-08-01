# Smart Refrigerator Keyword Spotting using CNN

A lightweight Keyword Spotting (KWS) system designed for resource-constrained smart refrigerators. This project applies Convolutional Neural Networks (CNNs) to recognize predefined voice commands using MFCC features while maintaining a compact model suitable for embedded devices.

---

## Project Overview

This project aims to develop a lightweight speech recognition system capable of recognizing predefined keywords for smart refrigerator voice control.

To improve robustness under real-world conditions, the model incorporates:

- MFCC feature extraction
- CNN-based keyword classification
- Data augmentation
- L2 Regularization
- Dropout
- Lightweight network architecture

The final model contains approximately **0.1 million parameters**, making it suitable for deployment on resource-limited devices.

---

## Features

- Lightweight 2-layer CNN architecture
- MFCC-based audio feature extraction
- Data augmentation
  - Noise injection
  - Time shifting
  - Frequency shifting
- L2 Regularization
- Dropout
- TensorFlow implementation
- Training and testing pipeline
- Model checkpoint saving
- Confusion matrix evaluation

---

## Model Architecture

Input

```
32 × 32 MFCC Spectrogram
```

↓

Conv Layer 1

- 3×3 Convolution
- 16 Filters
- ReLU
- Max Pooling

↓

Conv Layer 2

- 3×3 Convolution
- 32 Filters
- ReLU
- Max Pooling

↓

Fully Connected Layer

- 64 Neurons
- ReLU

↓

Output Layer

- Softmax
- 7 Classes

---

## Technologies

- Python
- TensorFlow (v1 Compatibility API)
- NumPy
- SciPy
- Scikit-learn
- Matplotlib

---

## Dataset

The dataset consists of MFCC representations of predefined voice commands.

Training data and testing data are stored as MATLAB `.mat` files.

```
MFCC/
└── mat_data_augmented/
    ├── train.mat
    └── test.mat
```

Each sample is represented as a **32 × 32 MFCC feature map**.

---

## Training

Run the following command:

```bash
python main.py
```

The training process includes:

- Mini-batch training
- Adam Optimizer
- Learning rate decay
- L2 Regularization
- Dropout

Model checkpoints are automatically saved to:

```
model/
```

---

## Performance

| Metric | Result |
|---------|--------|
| Test Accuracy | **97.36%** |
| Parameters | **~0.1M** |
| Optimizer | Adam |
| Batch Size | 64 |
| Learning Rate | 0.001 |

---

## Project Structure

```
.
├── main.py
├── dp_refined.py
├── load.py
├── model/
├── MFCC/
│   └── mat_data_augmented/
│       ├── train.mat
│       └── test.mat
└── README.md
```

---

## Future Improvements

Possible future work includes:

- Replace TensorFlow v1 with TensorFlow 2 / PyTorch
- Improve inference speed
- Quantization for embedded deployment
- Support larger vocabulary
- Deploy on Raspberry Pi or edge devices

---

## Publication

Accepted for publication at The 2nd International Conference on Artificial Intelligence, Virtual Reality and Interaction Design (AIVRD 2026).

## Author

Chen Jia

Monash University Malaysia

Bachelor of Computer Science
