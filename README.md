# AI Land Cover Classification V3

A deep-learning based land-cover classification system built with
PyTorch, ResNet50, FastAPI, and a JavaScript frontend.

The system classifies satellite images into 10 EuroSAT land-cover
categories and uses feature-space similarity to detect images that
may be outside the training domain.

---

## Overview

Traditional image classifiers can produce highly confident
predictions even when an input image is completely different from
the training data.

This project addresses that problem by combining:

1. ResNet50 image classification
2. Prediction confidence
3. Feature-space similarity
4. Out-of-Distribution (OOD) detection
5. Domain Fit scoring

The result is a system that does not only answer:

> "What class does the model predict?"

It also asks:

> "Does this image actually look like something the model was
> trained to recognize?"

---

## Features

- ResNet50-based land-cover classifier
- 10 EuroSAT land-cover classes
- Top-3 predictions
- Prediction confidence
- Prediction entropy
- Prediction margin
- 2048-dimensional deep feature extraction
- Class-centroid based feature similarity
- Feature-space OOD detection
- Domain Fit / Reliability score
- FastAPI REST API
- Interactive HTML/CSS/JavaScript frontend
- Drag-and-drop image upload
- Real-time prediction results
- API documentation through Swagger UI

---

## Supported Classes

The model predicts one of the following 10 EuroSAT classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

---

## Model

### Architecture

The classifier uses:

**ResNet50 pretrained on ImageNet**

The original classification head was replaced with:

```text
Linear(2048 → 10)
                    ┌─────────────────────┐
                    │      Frontend       │
                    │   HTML / CSS / JS   │
                    └──────────┬──────────┘
                               │
                               │ POST /predict
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ResNet50       │
                    │   Image Classifier  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        Classification               Feature Extraction
                 │                           │
                 │                     2048-D vector
                 │                           │
                 │                           ▼
                 │                  Cosine Similarity
                 │                           │
                 │                           ▼
                 │                      OOD Check
                 │                           │
                 │                           ▼
                 │                      Domain Fit
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                         JSON Response
                               │
                               ▼
                           Frontend