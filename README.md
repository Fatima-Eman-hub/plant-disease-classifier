# Plant Disease Classifier

A deep learning image classification system for identifying plant diseases from leaf images.

## Overview

This project compares a custom CNN with a pretrained ResNet18 model for plant disease classification.

The final ResNet18 model achieved approximately 99.56% validation accuracy.

## Features

- Plant disease image classification
- Custom CNN baseline
- ResNet18 transfer learning
- Data augmentation
- Model evaluation
- Error analysis
- Confidence calibration
- Top-5 predictions
- Interactive Gradio interface

## Model

ResNet18 with transfer learning.

## Performance

| Model | Validation Accuracy |
|---|---:|
| Custom CNN | 94.90% |
| ResNet18 | 99.56% |

Expected Calibration Error (ECE):

0.0012

## Usage

Upload a plant leaf image and the application returns:

- Predicted disease
- Prediction confidence
- Top-5 predictions

## Disclaimer

This project is intended for educational and research purposes. Predictions should not be treated as a professional agricultural diagnosis.