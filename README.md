# DeepFake-Video-Detection

This project presents a hybrid deepfake video detection system that combines the spatial feature extraction strength of a custom-designed XceptionNet with Swiss Residual Blocks and the temporal modeling capabilities of a ResNet-Swish-BiLSTM architecture. Built using the Meta DeepFake Detection Challenge (DFDC) dataset, our model processes video frames to detect both intra-frame manipulations and inter-frame inconsistencies. The architecture incorporates an attention mechanism to focus on the most relevant frames, while Focal Loss, label smoothing, and mixed precision training address class imbalance and enhance generalization. We implemented a carefully designed preprocessing pipeline using MTCNN for face detection, frame sampling for temporal coverage, and extensive data augmentation. The final model achieves a validation accuracy of 97.01% and a ROC-AUC of 0.988, outperforming standard models like vanilla Xception and BiLSTM while maintaining computational efficiency. This repository includes code for data preprocessing, model architecture, training routines, and an efficient inference pipeline for real-time fake video classification.

**Features**

📹 Input: Preprocessed face crops from equi-spaced sampled video frames (via MTCNN)

🧠 Architecture: Combines XceptionNet with Swiss Residual Blocks + 2-layer BiLSTM + Attention + Classifier

🎯 Accuracy: Achieved 97.01% validation accuracy and 0.988 ROC-AUC on DFDC dataset

⚙️ Optimizations: Used Focal Loss, label smoothing, transfer learning, and mixed precision training

📈 Robustness: Extensive data augmentation, dropout regularization, and frame-level attention for generalization


**🛠 Technologies Used**

PyTorch, OpenCV, MTCNN, timm, NumPy, Pandas

Focal Loss, Label Smoothing, Mixed Precision, CosineAnnealingWarmRestarts

DFDC Dataset (9GB subset)
