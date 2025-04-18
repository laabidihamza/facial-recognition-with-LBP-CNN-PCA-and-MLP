# 🧠 Face Recognition with LBP, PCA, and MLP

This project implements a facial recognition system using two different methods **Local Binary Patterns (LBP)** and **Convolutional Neural Network (CNN)** for feature extraction, **Principal Component Analysis (PCA)** for dimensionality reduction, and a **Multi-Layer Perceptron (MLP)** for classification. The system is trained and tested on the **FEI Face Database**.

---

## 🛠️ Requirements

```bash
pip install numpy opencv-python scikit-learn matplotlib keras tensorflow
```

---

## 📁 Dataset: FEI Face Database

- 200 subjects (100 male, 100 female)
- 14 images per person → **2800 total images**
- Controlled conditions: white background, upright frontal position, varying facial expressions and head rotations
- Image size: 640x480

---

## 🔍 Project Pipeline

1. **Data Loading**: Load grayscale face images from directories with filenames in `ID-XX.jpg` format.
2. **Preprocessing**: Resize images to `100x100`, convert to grayscale if needed.
3. **Feature Extraction**:  
   A. Use **LBP (Local Binary Patterns)** to capture local texture information.  
   - Convert LBP patterns to a **histogram of binary values**.

   B. A shallow **Convolutional Neural Network** architecture learns to extract meaningful spatial features from face images.  
   - Output of the CNN (flattened layer) is used as a feature vector.
4. **Dimensionality Reduction**:  
   - Apply **PCA** to reduce the high-dimensional LBP feature vector into lower dimensions.
5. **Classification**:  
   - Train an **MLP neural network** using reduced features.
   - Evaluate accuracy on a test set.
6. **Prediction**:  
   - Given a test image, extract LBP features → reduce via PCA → classify with MLP → return predicted identity.

---

## 📈 Visualizations

- 📉 **Loss and Accuracy Curves** during training  
- 📷 **Predicted Output Image** with recognized identity