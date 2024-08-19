# Fine-Tuning Named Entity Recognition with SpaCy

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SpaCy](https://img.shields.io/badge/SpaCy-3.0+-orange.svg)
## Overview

This project demonstrates how to fine-tune a pre-trained Named Entity Recognition (NER) model using SpaCy. The project involves loading a real-world dataset, annotating the text with named entities, and fine-tuning the SpaCy model to recognize specific entities. The goal is to customize the model for a particular domain and evaluate its performance on unseen data.

## Dataset

The dataset used in this project is a collection of annotated text data containing various named entities such as organizations, locations, persons, dates, etc. The dataset is split into training and test sets to evaluate the fine-tuned model's performance.

## Features

- **Loading and Preprocessing Data:** Load a dataset containing text and corresponding entity annotations.
- **Fine-Tuning a Pre-trained NER Model:** Use SpaCy to fine-tune an NER model on the provided dataset.
- **Evaluation:** Evaluate the fine-tuned model's performance on the test set.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/ner_finetuning_spacy.git
   ```
2. Run with:
   ```bash
   python fine-tuning.py
   ```
