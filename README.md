# Anomaly Detection Project

This project implements anomaly detection models, focusing on Credit Fraud detection using a Hybrid Score approach (Reconstruction Error + Latent Distance) with a Transformer-based VAE and Contrastive Learning. It also includes an Isolation Forest baseline.

## Project Structure

- `main.py`: Main entry point for training the Transformer VAE model and evaluating with the Hybrid Score.
- `iForest.py`: Baseline implementation using Isolation Forest.
- `DataLoader.py`: Handles dataset loading and preprocessing for various datasets (Fraud, NASA, Telco, Cora, etc.).
- `dataset.py`: PyTorch Dataset definition and data splitting logic.
- `model.py`: Transformer VAE model architecture and loss function.
- `visualize.py`: Visualization utilities (presumed).

## Requirements

The project requires the following Python packages:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `torch_geometric`
- `requests`
- `torchvision`

## Usage

### 1. Train and Evaluate Hybrid Model

To train the Transformer VAE model and evaluate it using the Hybrid Score method:

```bash
python main.py
```

This will:
1. Load the Credit Fraud dataset.
2. Train the TransformerVAE model.
3. Calculate the "Normal Center" in the latent space.
4. Evaluate using the Hybrid Score (Reconstruction Error + Euclidean Distance to Normal Center).
5. Output AUROC, AUPRC, and Best F1-Score.

### 2. Run Baseline (Isolation Forest)

To run the Isolation Forest baseline:

```bash
python iForest.py
```

## Datasets

The `DataLoader.py` supports multiple datasets. The default for `main.py` is `fraud` (Credit Fraud). Other supported datasets include:
- `nasa` (CMAPSS)
- `telco` (Customer Churn)
- `cora` (Citation Network)
- `movielens`
- `fmnist` (Fashion MNIST)
- `mnist`
- `news`
