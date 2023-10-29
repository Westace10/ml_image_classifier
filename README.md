# Flower Image Classification

## Overview

This project is a flower image classification system that uses a neural network to classify different species of flowers.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

The project consists of two main components: training and prediction.

- **Training**: This component trains a neural network using flower images to classify different flower species.

- **Prediction**: This component uses a pre-trained model to predict the flower species of a given image.

## Prerequisites

Before you get started, make sure you have the following prerequisites installed on your system:

- Python 3
- PyTorch
- torchvision
- Pillow (PIL)

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a model on flower images, you can use the `train.py` script. Here are the available options:

- `data_directory` (required): The path to the data directory containing the flower images.
- `--save_dir` (optional): The directory to save checkpoints (default is "checkpoints").
- `--arch` (optional): The architecture to use (vgg16 or densenet121, default is "vgg16").
- `--learning_rate` (optional): The learning rate (default is 0.001).
- `--hidden_units` (optional): The number of hidden units in the classifier (default is 512).
- `--epochs` (optional): The number of training epochs (default is 10).
- `--gpu` (optional): Use GPU for training (flag).

Example:

```bash
python train.py /path/to/data/directory --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

### Prediction

To predict the flower species of an image, you can use the `predict.py` script. Here are the available options:

- `input` (required): The path to the input image for prediction.
- `checkpoint` (required): The path to the model checkpoint.
- `--top_k` (optional): Return the top K most likely classes (default is 5).
- `--category_names` (optional): Mapping of categories to real names (default is "cat_to_name.json").
- `--gpu` (optional): Use GPU for inference (flag).

Example:

```bash
python predict.py /path/to/input/image /path/to/model/checkpoint --top_k 5 --category_names cat_to_name.json --gpu
```

## Project Structure

The project is structured as follows:

- `train.py`: Script for training the model.
- `predict.py`: Script for predicting the flower species of an image.
- `model_utils.py`: Utility functions for model training and prediction.
- `requirements.txt`: List of required packages.
- `cat_to_name.json`: Mapping of categories to real names.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the [Udacity](https://www.udacity.com/) team for providing the project guidelines and data.
- Flower images dataset sourced from [source](https://example.com).

