# Visual Nutrition Estimation

This repository implements the **DPF-Nutrition** model, based on the paper [DPF-Nutrition: Food Nutrition Estimation via Depth Prediction and Fusion](https://doi.org/10.3390/foods12234293), with minor modifications. The model has been trained and fine-tuned on overhead images from Google’s [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k) to estimate nutritional information from images of food items.

## Inference

To run inference, please refer to:
- **[inference.py](inference.py)**: Script-based inference
- **[inference.ipynb](inference.ipynb)**: Notebook-based inference

## Input Image Requirements

For optimal performance, input images should meet the following requirements:

- **Format:** RGB
- **Resolution:** 640 x 480 pixels
- **Pixel Range:** 0 to 255 (values do not need to reach exactly 255 but should be within this range)

### Preprocessing Transformation

Use this transformation to preprocess your images:

```python
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0) 
])
```

## Model Weights

Download the pre-trained model weights from the links below:

1. **DPT Model Weights (dpt.py):** [Google Drive Link](https://drive.google.com/file/d/1uKLE2oQmmfVUN3PthgVlIujVMrTyoACV/view?usp=sharing)
2. **DPF-Nutrition Model Weights (dpfnutrition.py):** [Google Drive Link](https://drive.google.com/file/d/1b8bJbLJMUd6vz_V5NyYCr1HgSCRsCkck/view?usp=sharing)