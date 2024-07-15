# Stress Identification System

This project is a machine learning-based system designed to identify stress in text data. It uses several classifiers including Support Vector Machine (SVM), Random Forest, and Naive Bayes to predict whether a given text indicates stress or not.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Description

The Stress Identification System leverages Natural Language Processing (NLP) techniques to analyze textual data and classify it as "stressed" or "non-stressed". The system employs three different classifiers (SVM, Random Forest, and Naive Bayes) to provide accurate predictions.

## Installation

To get started with this project, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/stress-identification-system.git
    ```

2. Navigate to the project directory:
    ```bash
    cd stress-identification-system
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, you need to have your dataset in a CSV file format with `TEXT DATA` and `LABEL` columns.

### Dataset
The dataset used for training and testing the models consists of text data labeled as either "stressed" (1) or "non-stressed" (0). Make sure your dataset is in CSV format with the following columns:

## TEXT DATA:
The text data to be analyzed.
## LABEL: 
The label indicating stress (1 for stressed, 0 for non-stressed).

### Model Architecture
## Support Vector Machine (SVM)
Kernel: Linear
Vectorizer: TF-IDF
## Random Forest
Number of Estimators: 100
Random State: 42
Vectorizer: TF-IDF
## Naive Bayes
Model: Multinomial Naive Bayes
Vectorizer: CountVectorizer


## Results
The performance of the models is evaluated using accuracy, classification report, and confusion matrix. 

## License
This project is licensed under the MIT License - see the LICENSE file for details.
