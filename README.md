# Visual Inspection Demo

## Computer Vision in Visual Inspection:
• Automated inspection ensuring 24/7 consistency and accuracy  
• Real-time defect detection reducing production losses  
• Scalable solution for high-volume inspection needs  
• Data-driven insights for process improvement

## Overview
This is a Streamlit application demonstrating four visual inspection use cases using computer vision models trained on the Clarifai Platform:

1. **Anomaly Detection**
   - Detects product anomalies with heatmap visualization

2. **Insulator Defect Detection**
   - Identifies defects in electrical insulators
   - Provides bounding boxes and confidence scores

3. **Crack Segmentation**
   - Segments and highlights cracks in surfaces
   - Real-time crack region visualization

4. **Surface Defect Detection**
   - Detects defects in metal surfaces
   - Classification with confidence scores

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuration
- Customize via sidebar controls
- Adjustable thresholds and display options
- Configurable image sources
- Visual theme customization

## Models
All models used are custom-trained using the Clarifai platform and tailored to each specific use case. The Clarifai platform simplifies the entire process of creating and training AI models, making it both easy and efficient. With just a single click, your model is not only trained but also automatically deployed, ready to enhance your business solutions instantly.

- Anomaly Detection (pill-anomaly)
- Insulator Detection (insulator-condition-inception)
- Crack Segmentation (crack-segmentation)
- Surface Defect Detection (surface-defects)