# Plant Disease Classifier

An end-to-end computer vision project that classifies plant leaf images into disease and healthy classes. The trained ResNet18 model is served through a lightweight Streamlit app with top-5 predictions and confidence scores.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://plant-disease-classifier-01.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Live Demo

Try the deployed application: **[plant-disease-classifier-01.streamlit.app](https://plant-disease-classifier-01.streamlit.app)**

Upload a JPG, JPEG, or PNG image of a leaf. The app returns the most likely class, confidence, and the five highest-probability predictions.

## Screenshots

### Upload an image

![Plant disease classifier upload screen](UI%20Images/IMG1.png)

### Uploaded leaf

![Uploaded leaf image](UI%20Images/IMG2.png)

### Prediction results

![Top prediction and confidence chart](UI%20Images/IMG3.png)

## Results

| Model | Validation accuracy |
| --- | ---: |
| Custom CNN baseline | 94.90% |
| ResNet18 transfer learning | **99.56%** |

The ResNet18 model improved validation accuracy by **4.66 percentage points** over the custom CNN baseline.

### Calibration

The final model achieved an Expected Calibration Error (ECE) of **0.0012**, indicating that its confidence scores were closely aligned with observed validation accuracy.

### Error analysis

The accompanying notebook contains the model evaluation workflow, including:

- Classification report and per-class metrics
- Confusion matrix inspection
- Misclassification review
- Confidence calibration analysis

## System Architecture

```mermaid
flowchart LR
	A[Leaf image upload] --> B[RGB conversion]
	B --> C[Resize to 224 x 224]
	C --> D[ImageNet normalization]
	D --> E[ResNet18 feature extractor]
	E --> F[Linear classifier]
	F --> G[Softmax probabilities]
	G --> H[Top prediction]
	G --> I[Top-5 results]
```

## Model Pipeline

1. Load the trained ResNet18 checkpoint.
2. Replace the original classification head with a linear layer matching the PlantVillage class count.
3. Convert each uploaded image to RGB.
4. Resize and normalize it using ImageNet statistics.
5. Run inference with gradients disabled.
6. Return the top prediction and top-five probability ranking.

## Repository Structure

```text
.
├── app.py                              # Streamlit inference application
├── plantvillage_resnet18_final.pth     # Trained ResNet18 checkpoint
├── plantvillage_class_names.json       # Class index to label mapping
├── plant_disease_dataset.csv           # Dataset metadata
├── DL(Plant_Disease_Classifier).ipynb  # Training and evaluation notebook
├── plant_images(FOR TESTING)/          # Local test images
├── UI Images/                          # Application screenshots
└── requirements.txt                    # Python dependencies
```

## Run Locally

```bash
git clone https://github.com/Fatima-Eman-hub/plant-disease-classifier.git
cd plant-disease-classifier
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

The app is configured for Streamlit Community Cloud:

1. Open [share.streamlit.io](https://share.streamlit.io/).
2. Select this GitHub repository and the `main` branch.
3. Set the main file to `app.py`.
4. Click **Deploy**.

## Disclaimer

This project is intended for educational and research purposes. Predictions should not be treated as a professional agricultural diagnosis. Image quality, lighting, background, and plant variety can affect results.