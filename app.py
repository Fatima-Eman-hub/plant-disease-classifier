
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import gradio as gr

from torchvision import models, transforms


# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# PATHS
# ============================================

MODEL_PATH = "plantvillage_resnet18_final.pth"
CLASS_NAMES_PATH = "plantvillage_class_names.json"


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

with gr.Blocks(
    title="Plant Disease Classifier"
) as demo:

    gr.Markdown(
        """
        # Plant Disease Classifier

        Upload a leaf image to classify the plant disease.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Leaf Image"
            )

            predict_button = gr.Button(
                "Predict",
                variant="primary"
            )

        with gr.Column():

            prediction_output = gr.Label(
                label="Predictions",
                num_top_classes=5
            )

            confidence_output = gr.Textbox(
                label="Top Prediction",
                interactive=False
            )

    gr.Markdown("### Top Predictions")

    results_table = gr.Dataframe(
        headers=[
            "Rank",
            "Class",
            "Confidence"
        ],
        datatype=[
            "number",
            "str",
            "str"
        ],
        interactive=False
    )

    predict_button.click(
        fn=predict_leaf,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output,
            results_table
        ]
    )


# ============================================
# LAUNCH
# ============================================

if __name__ == "__main__":
    demo.launch(share=True)
