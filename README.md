## 🔬 Data Processing & Normalization

In this project, **data preprocessing** is optimized for **energy-efficient model training**.

### **✅ Normalization（正規化）**
- **Image Pixel Scaling**:  
  - **[-1,1] Scaling**: Applied to image datasets to improve network stability.
  - Formula:  
    \[
    X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
    \]
  - Implemented in [`data_processing.py`](src/data_processing.py).

### **✅ Standardization（標準化）**
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

🚀 **Implemented in**: [`data_processing.py`](src/data_processing.py)
