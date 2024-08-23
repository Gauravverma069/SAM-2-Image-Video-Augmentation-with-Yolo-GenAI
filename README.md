# Image and Video Augmentation with Meta SAM 2 and YOLO Models

This project leverages the **Meta SAM 2 model** for advanced masking and mask propagation in both images and videos. It integrates two **YOLO models** for object detection, an **Image Annotator** for generating YOLO-compatible annotations, and various augmentation techniques, including generative AI capabilities. Additionally, users can manually enter frame numbers and object coordinates if they prefer not to use the YOLO models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [YOLO Object Detection](#yolo-object-detection)
  - [Model 1: Trainable YOLO](#model-1-trainable-yolo)
  - [Model 2: Pretrained YOLO](#model-2-pretrained-yolo)
- [Manual Entry of Frame Numbers and Coordinates](#manual-entry-of-frame-numbers-and-coordinates)
- [Meta SAM 2 Operations](#meta-sam-2-operations)
  - [Masking](#masking)
  - [Inverse Masking](#inverse-masking)
  - [Pixelation](#pixelation)
  - [Hue Change](#hue-change)
  - [Mask Replacement](#mask-replacement)
  - [Glow Effect](#glow-effect)
  - [Image-to-Image Generation](#image-to-image-generation)
- [Image Annotator Application](#image-annotator-application)
- [Deployment](#deployment)
- [Usage](#usage)
- [UseCases](#usecases)

## Overview

This project implements a comprehensive pipeline for augmenting images and videos. By combining the **Meta SAM 2 model** for masking and two **YOLO models** for object detection, the project offers various augmentation techniques, including pixelation, hue change, mask replacement, and glow effects. Additionally, it supports generative AI-driven image-to-image transformation using the **Stability AI API**.

## Features

- **Meta SAM 2 Model**: Handles masking and mask propagation.
- **YOLO Object Detection**:
  - Trainable YOLO model with built-in augmentation.
  - Pretrained YOLO model for quick object detection.
- **Manual Entry**: Option to manually enter frame numbers and coordinates for object detection.
- **Image Annotator Application**: Generates YOLO-compatible annotation `.txt` files.
- **Augmentation Techniques**:
  - Masking, inverse masking, pixelation, hue change, mask replacement, and glow effect.
  - Generative AI-driven image-to-image transformation.

## YOLO Object Detection

### Model 1: Trainable YOLO

- Allows user to train a custom YOLO model with input images.
- Performs image augmentation for enhanced training.

### Model 2: Pretrained YOLO

- Uses a pretrained YOLO model.
- Requires only the object name as input.
- Outputs frame number and object coordinates, feeding them into the SAM 2 model.

## Manual Entry of Frame Numbers and Coordinates

If you prefer not to use the YOLO models, you can manually enter the frame numbers and coordinates for the objects. This input is directly fed into the SAM 2 model for further processing.

## Meta SAM 2 Operations

### Masking

- Detects and masks objects identified by the YOLO models or through manual input.

### Inverse Masking

- Masks the area outside the detected object.

### Pixelation

- Applies pixelation to the masked area.

### Hue Change

- Adjusts the hue of the masked area.

### Mask Replacement

- Replaces the masked area with another image or video content.

### Glow Effect

- Adds a glowing effect to the masked area.

### Image-to-Image Generation

- Transforms the masked area using Stability AI's generative models.

## Image Annotator Application

The Image Annotator application allows for manual annotation of images to generate YOLO-compatible `.txt` files, streamlining the training process for custom object detection models.

## Deployment

- **Streamlit Application**: The project is deployed as a Streamlit application on Hugging Face Spaces.
- **Deployment URL**: [Huggingface deployment](https://huggingface.co/spaces/Gaurav069/SAM_2_Image_Augmentation_with_Yolo_GenAI)
- **Model URL**: [SAM 2 Model](https://huggingface.co/spaces/Gaurav069/SAM_2_Image_Augmentation_with_Yolo_GenAI/blob/main/sam2_hiera_base_plus.pt)
- **Video URL**: [Videos of Deployment](https://huggingface.co/spaces/Gaurav069/SAM_2_Image_Augmentation_with_Yolo_GenAI/tree/main/images)

## Usage

1. Train the YOLO model or use the pretrained YOLO model.
2. Optionally, manually input frame numbers and coordinates.
3. Use the Image Annotator for custom annotations.
4. Run the augmentation pipeline with the SAM 2 model for desired operations.

## UseCases

- Visit Use cases page on streamlit application.

