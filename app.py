
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import streamlit as st
from PIL import Image

from torchvision import models, transforms


# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# PATHS
# ============================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "plantvillage_resnet18_final.pth"
CLASS_NAMES_PATH = APP_DIR / "plantvillage_class_names.json"


# ============================================
# LOAD CLASS NAMES
# ============================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)


# ============================================
# LOAD RESNET18
# ============================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================
# IMAGE PREPROCESSING
# ============================================

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================
# PREDICTION
# ============================================

def predict_leaf(image):

    if image is None:
        return {}, "Please upload a leaf image.", []

    image = image.convert("RGB")

    image_tensor = inference_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    top_k = min(5, len(class_names))

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k
    )

    predictions = {
        class_names[index.item()]:
        float(probability)

        for probability, index
        in zip(
            top_probabilities,
            top_indices
        )
    }

    predicted_index = top_indices[0].item()

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        top_probabilities[0].item() * 100
    )

    top_prediction = (
        f"{predicted_class} — "
        f"{confidence:.2f}%"
    )

    table_data = []

    for rank, (probability, index) in enumerate(
        zip(
            top_probabilities,
            top_indices
        ),
        start=1
    ):

        table_data.append([
            rank,
            class_names[index.item()],
            f"{probability.item() * 100:.2f}%"
        ])

    return (
        predictions,
        top_prediction,
        table_data
    )


# ============================================
# USER INTERFACE
# ============================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿"
)

st.title("Plant Disease Classifier")
st.write("Upload a leaf image to classify the plant disease.")

uploaded_file = st.file_uploader(
    "Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    if st.button("Predict", type="primary"):
        predictions, top_prediction, table_data = predict_leaf(image)

        st.subheader("Top Prediction")
        st.write(top_prediction)

        st.subheader("Predictions")
        st.bar_chart(predictions)

        st.subheader("Top Predictions")
        st.dataframe(
            table_data,
            column_config={
                0: "Rank",
                1: "Class",
                2: "Confidence"
            },
            hide_index=True,
            use_container_width=True
        )
