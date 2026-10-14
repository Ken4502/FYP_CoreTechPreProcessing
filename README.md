# 🌿 Plant Image Preprocessing Studio

An interactive preprocessing application that filters plants from background clutter and noise, producing two specialized **cropped plant outputs**:

1. **Method 1: Cropped Plant for Plant Species Identification**
2. **Method 2: Cropped Plant for Disease & Health Condition Identification**

*(Note: This application focuses strictly on high-precision image preprocessing and background removal without running classification models).*

---

## 🔬 The Two Preprocessing Methods

| Feature | Method 1: Plant Species Identification | Method 2: Disease & Health Identification |
| :--- | :--- | :--- |
| **Objective** | Preserve overall plant morphology, leaf margin serrations, apex, and petiole structure. | Preserve genuine lesion colors (chlorosis halos, necrosis), surface spore textures, and prevent internal holes. |
| **Color Treatment** | **Gray-World White Balancing**: Standardizes natural foliage colors and removes environmental sunlight/shadow color casts. | **Genuine Color Retention**: Keeps raw chromatic hues intact so yellow chlorosis and brown necrotic tissue are not washed out. |
| **Lamina Protection** | Crisp organic boundary delineation. | **Zero-Hole Solid Lamina**: Guarantees that necrotic and fungal spots inside the leaf are 100% enclosed and never cut out. |
| **Surface Enhancement** | Natural texture without artificial contrast. | **Selective Foreground CLAHE**: Enhances subtle fungal pustules, concentric rings, and lesion borders on the leaf surface. |
| **Cropping Geometry** | **Square Canvas with Letterbox Padding**: Preserves the leaf's biological aspect ratio (prevents stretching). | **Tight Symptomatic Bounding Box**: Zooms in on the foliage to maximize pixel density of disease spots. |

---

## 🛠️ Quick Start

### 1. Run the Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. How to Use
1. **Select or Upload**: Upload any plant image (JPG, PNG, WEBP) or test with the preset benchmark samples.
2. **Review the Results**: Instantly view the two cropped outputs side-by-side.
3. **Download**: Click the download button below either output to save the clean cropped PNG image.
