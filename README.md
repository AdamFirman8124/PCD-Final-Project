# Apple Leaf Disease Detection

This project implements a computer vision workflow for detecting diseases in apple leaves. The main workflow consists of:

1. **Segmentation**: Separating leaf from background and identifying healthy vs. diseased regions
2. **Feature Extraction**: Computing morphological and texture features from the segmented regions
3. **SVM Classification**: Training an SVM model to classify leaves as healthy or diseased

## Dataset

The dataset contains apple leaf images categorized into:
- Healthy leaves
- Apple scab
- Black rot
- Cedar apple rust

## Requirements

Install the required packages:

```
pip install -r requirements.txt
```

## How to Run

Simply execute the main script:

```
python apple_leaf_disease_detection.py
```

## Workflow Details

### Segmentation
- Converts images to HSV color space
- Uses color thresholds to identify healthy parts (green) and diseased parts (yellow/brown)
- Applies morphological operations to clean up segmentation

### Feature Extraction
- **Morphological Features**: Area, perimeter, eccentricity, extent, solidity, disease ratio
- **Texture Features**: GLCM-based features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
  - Extracted separately for healthy and diseased regions

### SVM Model
- Uses RBF kernel for classification
- Evaluates model using classification report and confusion matrix

## Results

Results are saved in the `Results` directory, including:
- Feature CSV file
- Visualization images showing:
  - Original image
  - Segmented leaf
  - Color-coded visualization (green = healthy parts, red = diseased parts)

## Example Visualization

The visualizations show:
1. The original leaf image
2. The segmented leaf regions
3. The detected healthy (green) and diseased (red) areas with classification results 