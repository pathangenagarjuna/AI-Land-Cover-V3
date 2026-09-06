import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

NUM_CLASSES = 10

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# Frozen threshold selected using validation set
OOD_THRESHOLD = 0.42


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "model" /
    "v2_best_landcover_model.pth"
)

CENTROID_PATH = (
    BASE_DIR /
    "model" /
    "v3_class_centroids.pt"
)

VALIDATION_REFERENCE_PATH = (
    BASE_DIR /
    "model" /
    "v3_validation_similarity_reference.pt"
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD RESNET50
# =========================================================

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =========================================================
# FEATURE EXTRACTOR
# =========================================================

class FeatureExtractor(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,

            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,

            model.avgpool
        )

    def forward(self, x):

        x = self.backbone(x)

        x = torch.flatten(
            x,
            1
        )

        return x


feature_model = FeatureExtractor(
    model
).to(device)

feature_model.eval()


# =========================================================
# LOAD CLASS CENTROIDS
# =========================================================

centroid_matrix = torch.load(
    CENTROID_PATH,
    map_location=device
)

centroid_matrix = centroid_matrix.to(device)

centroid_matrix = F.normalize(
    centroid_matrix,
    dim=1
)


# =========================================================
# LOAD VALIDATION SIMILARITY REFERENCE
# =========================================================

validation_similarities = torch.load(
    VALIDATION_REFERENCE_PATH,
    map_location="cpu"
)

validation_similarities = (
    validation_similarities
    .detach()
    .cpu()
    .numpy()
)

validation_similarities = np.sort(
    validation_similarities
)


# =========================================================
# DOMAIN RELIABILITY SCORE
# =========================================================

def calculate_reliability_score(similarity, is_ood):
    """
    Domain-fit score from 0 to 100.

    This is NOT the probability that the prediction
    is correct.

    It represents how well the image's feature similarity
    fits the EuroSAT validation distribution.

    Possible OOD images receive a score of 0.
    """

    # Anything below the validated OOD threshold
    # is considered low domain fit.
    if is_ood:
        return 0.0

    # Percentile within EuroSAT validation distribution
    percentile = (
        np.searchsorted(
            validation_similarities,
            similarity,
            side="right"
        )
        / len(validation_similarities)
    ) * 100

    percentile = np.clip(
        percentile,
        0,
        100
    )

    return round(
        float(percentile),
        1
    )

# =========================================================
# RELIABILITY STATUS
# =========================================================

def get_reliability_status(reliability_score,is_ood):

    if is_ood:
        return "Low Reliability"

    if reliability_score >= 70:
        return "High Reliability"

    if reliability_score >= 40:
        return "Medium Reliability"

    return "Low Reliability"


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # -----------------------------------------------------
    # Convert to RGB
    # -----------------------------------------------------

    image = image.convert("RGB")


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)


    # =====================================================
    # MODEL INFERENCE
    # =====================================================

    with torch.no_grad():

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


        # -------------------------------------------------
        # Top 3
        # -------------------------------------------------

        values, indices = torch.topk(
            probabilities,
            3
        )


        prediction_index = (
            indices[0].item()
        )

        prediction = (
            CLASS_NAMES[
                prediction_index
            ]
        )

        confidence = (
            values[0].item()
        )


        # -------------------------------------------------
        # Entropy
        # -------------------------------------------------

        entropy = -(
            probabilities *
            torch.log(
                probabilities + 1e-10
            )
        ).sum().item()


        normalized_entropy = (
            entropy /
            np.log(NUM_CLASSES)
        )


        # -------------------------------------------------
        # Prediction margin
        # -------------------------------------------------

        margin = (
            values[0] -
            values[1]
        ).item()


        # -------------------------------------------------
        # Feature extraction
        # -------------------------------------------------

        feature = feature_model(
            tensor
        )

        feature = F.normalize(
            feature,
            dim=1
        )


        # -------------------------------------------------
        # Similarity with all centroids
        # -------------------------------------------------

        similarities = torch.mm(
            feature,
            centroid_matrix.T
        )[0]


        # -------------------------------------------------
        # Nearest centroid similarity
        # -------------------------------------------------

        feature_similarity, nearest_index = (
            torch.max(
                similarities,
                dim=0
            )
        )

        feature_similarity = (
            feature_similarity.item()
        )

        nearest_class = (
            CLASS_NAMES[
                nearest_index.item()
            ]
        )


    # =====================================================
    # OOD DECISION
    # =====================================================

    is_ood = (
        feature_similarity <
        OOD_THRESHOLD
    )


    # =====================================================
    # RELIABILITY
    # =====================================================

    reliability_score = calculate_reliability_score(
        feature_similarity,
        is_ood
    )   

    reliability_status = (
        get_reliability_status(
            reliability_score,
            is_ood
        )
    )


    # =====================================================
    # TOP 3 RESULT
    # =====================================================

    top3 = []

    for value, index in zip(
        values,
        indices
    ):

        top3.append({
            "class": CLASS_NAMES[
                index.item()
            ],

            "probability": round(
                value.item() * 100,
                2
            )
        })


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "prediction": prediction,

        "confidence": round(
            confidence * 100,
            2
        ),

        "top3": top3,

        "entropy": round(
            normalized_entropy,
            4
        ),

        "margin": round(
            margin * 100,
            2
        ),

        "feature_similarity": round(
            feature_similarity,
            4
        ),

        "nearest_eurosat_class": (
            nearest_class
        ),

        "ood": is_ood,

        "ood_threshold": OOD_THRESHOLD,

        "reliability_score": (
            reliability_score
        ),

        "reliability_status": (
            reliability_status
        )
    }


# =========================================================
# STARTUP INFORMATION
# =========================================================

print("V3 predictor loaded successfully.")

print( "Device:", device )

print( "OOD threshold:",OOD_THRESHOLD)

print( "Validation reference samples:", len(validation_similarities))