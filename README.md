# ⚡ Efficient Deep Learning: Optimizing Energy Consumption in AI Training

This project explores **energy-efficient deep learning** by optimizing **training cost, accuracy, and computational resources**. Using **ResNet architectures, FFNN architectures, Xavier initialization, and pruning techniques**, we investigate how model depth, width, and learning rate impact **cumulative energy consumption** while maintaining high accuracy.

---

## 🚀 Project Overview

- **Objective**: Reduce energy consumption in deep learning training without compromising accuracy.
- **Model**: ResNet with **shortcut connections** & **Xavier initialization**
- **Techniques**: Pruning, Learning Rate Optimization, Width/Depth Trade-off Analysis
- **Key Findings**:
  - **Pruning + ResNet + Xavier reduces compute by 15-30%**
  - **Optimal learning rate (0.001) improves efficiency**
  - **Shortcut connections accelerate convergence**
- **Potential Applications**: **Edge AI, IoT, and low-power AI systems**

---

## 📂 Project Structure

| File/Folder            | Description                                  |
|------------------------|----------------------------------------------|
| 📂 src                | Core scripts for training and evaluation     |
| ├── train.py          | Train ResNet with Xavier initialization      |
| ├── evaluate.py       | Evaluate model accuracy & energy consumption |
| ├── energy_estimation.py | Compute cumulative energy consumption  |
| ├── prune.py         | Pruning techniques to reduce computation     |
| ├── data_processing.py | Normalize & standardize dataset           |
| ├── plot_results.py   | Generate accuracy-energy tradeoff plots     |
| 📂 experiments                 | Model experiments & evaluation |
| 📂 visualizations     | Stores training plots & energy consumption charts |
| 📄 ResNet_Energy_Efficiency_Pruning.ipynb | Full training pipeline notebook |
| 📄 requirements.txt   | Python dependencies                         |
| 📄 README.md          | Project documentation                       |

---

## 🔬 Data Processing & Normalization

In this project, **data preprocessing** is optimized for **energy-efficient model training**.

### **✅ Normalization **
- **Image Pixel Scaling**:  
  - **[-1,1] Scaling**: Applied to image datasets to improve network stability.
  - Formula:  
    \[
    X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
    \]
  - Implemented in [`data_processing.py`](src/data_processing.py).

### **✅ Standardization **
- **Zero Mean, Unit Variance Transformation**:  
  - Applied to structured input features to ensure stable gradient updates.
  - Formula:  
    \[
    X' = \frac{X - \mu}{\sigma}
    \]
  - Used for non-image datasets.

### **✅ Energy Consumption Data Handling**
- Computed per **input-hidden and hidden-output layers** to track efficiency.
- Accumulated energy usage analyzed across **training epochs**.

🚀 **Implemented in**: [`energy_estimation.py`](src/energy_estimation.py)

---

## 🏗 Model Architecture

The **ResNet-based model** is optimized for efficient deep learning training:

```python
model = ResNet(depth=34, shortcut=True, initialization='xavier')
```

### **Why ResNet?**
✅ **Shortcut connections** reduce gradient vanishing  
✅ **Xavier initialization** improves weight scaling  
✅ **Pruning techniques** remove redundant computations  

---

## **🔥 FFNN vs. ResNet Energy Consumption**
Our study compared **Feedforward Neural Networks (FFNN)** and **ResNet** in terms of **computational cost**.

| Model         | Energy Efficiency | Computational Cost |
|--------------|-----------------|-------------------|
| **FFNN (3-layer)** | ❌ High energy   | 🔺 High FLOPs     |
| **ResNet (No Pruning)** | ⚠️ Medium energy | 🔹 Moderate FLOPs |
| **ResNet + Pruning** | ✅ Low energy   | 🔻 Low FLOPs      |

🚀 **Implemented in**: [`train.py`](src/train.py)  

---

## 🎯 Training Strategy

🚀 **Key Optimizations**  
- **Optimizer**: **Adam** (efficient gradient updates)  
- **Loss Function**: **Softmax Cross-Entropy** (for multi-class classification)  
- **Training Setup**:  
  - **50 epochs**, batch size = **64, stop when acc = 98% for better comparison**  
  - **Pruning reduces compute by 30%**  
  - **Optimal learning rate 0.001** 

🔹 **Energy-Aware Pruning Strategy**  
- Pruning every 5 epochs → reduces 30% FLOPs, maintains 98% accuracy.
  

🚀 **Implemented in**: [`prune.py`](src/prune.py)  

---

## 📉 Training Performance  

✅ **Finding the Optimal Learning Rate**  
| **Learning Rate vs. Accuracy** | **Learning Rate vs. Energy Consumption** | **Learning Rate vs. Epochs** |
|------------------|-----------------------------|-----------------------------|
| ![LR vs. Accuracy](visualizations/lr_vs_accuracy.png) | ![LR vs. Energy](visualizations/lr_vs_energy.png) | ![LR vs. Epochs](visualizations/lr_vs_epochs.png) |

✅ **FFNN Width vs. Depth Impact on Energy and Epochs**  
| **FFNN Depth vs. Energy and Epochs** |
|-------------------|
| ![FFNN Depth](visualizations/ffnn_depth_vs_energy.png) | 

| **FFNN Width vs. Energy and Epochs** |
|-------------------|
| ![FFNN Width](visualizations/ffnn_width_vs_energy.png) |

🚀 **Implemented in**: [`train.py`](src/train.py)  


---

 

## 📈 Model Performance & Energy Efficiency  

✅ **Energy-Accuracy Tradeoff**: Shows the relationship between computational cost and model accuracy  
✅ **Depth vs. Energy Consumption**: Determines the optimal architecture for energy efficiency  
✅ **Pruning Impact **: Demonstrates how Pruning affects FLOPs and power consumption.

### **Energy vs Accuracy Tradeoff**
| **FFNN Depth vs. Energy Consumption** | **ResNet Depth vs. Energy Consumption** |
|-------------------|-------------------|
| ![FFNN Energy](visualizations/ffnn_depth_vs_energy2.png) | ![ResNet Energy](visualizations/resnet_depth_vs_energy.png) |



| **No Pruning Energy Consumption (left) vs. FFNN Pruning (right)** |
|-------------------|
| ![Pruning Impact](visualizations/ffnn_pruning_vs_no_pruning_energy.png) |

🚀 **Implemented in**: [`evaluate.py`](src/evaluate.py)  


---

## 🔥 Key Takeaways  

📌 **Best Configuration**:  
- **ResNet + Xavier + Pruning** achieved **15-30% compute reduction** while maintaining accuracy.  
- **Optimal Learning Rate (0.001) improves convergence & efficiency**.  
- **ResNet Shortcut connections accelerate training and reduce energy cost**.  

📌 **Future Directions**:  
- 🔹 **Explore alternative architectures (e.g., EfficientNet, Hybrid CNN-RNNs) for better trade-offs**.  
- 🔹 **Investigate model quantization (TensorFlow Lite, ONNX) for real-time deployment**.  
- 🔹 **Test performance on edge devices (e.g., Raspberry Pi, Jetson Nano) for practical applications**.  

---

## 🌍 Why It Matters  

AI-driven deep learning models are becoming increasingly complex, leading to **high computational cost and energy consumption**. This research contributes to **Green AI**, focusing on:  

- 🌱 **Reducing the carbon footprint of AI models**.  
- ⚡ **Improving energy efficiency for deep learning training**.  
- 📶 **Enabling low-power AI applications in IoT & Edge AI**.  

🚀 **This project paves the way for more sustainable AI solutions in real-world scenarios!**
