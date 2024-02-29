# U-Net Gap Filling for Satellite Imagery

This repository contains code for performing gap filling in satellite imagery using the U-Net architecture. The model is initially trained on a Landsat 4 (L4) gap-free dataset and then fine-tuned using data containing gaps.

## Overview

The U-Net architecture is a convolutional neural network (CNN) commonly used for image segmentation tasks. In this project, we utilize it to fill gaps in satellite imagery caused by cloud cover or other factors.

## Data

- **Landsat 4 (L4) Gap-Free Dataset**: Initially, the model is trained on a dataset with no gaps to learn features from clean imagery.
- **Gap Data**: After obtaining the pre-trained model, it is fine-tuned using data containing gaps to specifically address the task of filling missing regions in satellite images.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git


2. Clone the repository and go to the demo directory, then follow instruction there:
