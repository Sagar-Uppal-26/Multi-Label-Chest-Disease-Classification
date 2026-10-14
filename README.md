# Multi-Label Chest Disease Classification

A deep learning project that predicts 14 chest disease labels from chest X-ray images using PyTorch. It compares three architectures and uses Grad-CAM to visualize image regions influencing model predictions.

## Models

- **DenseNet-121** — pretrained CNN fine-tuned for multi-label classification.
- **EfficientNet-B0** — pretrained CNN for comparison.
- **Hybrid CNN-Transformer** — ResNet-34 feature extractor combined with a Transformer encoder.

## Tech Stack

Python, PyTorch, Torchvision, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib.

## Dataset

[NIH Chest X-ray 14 — 224×224 Resized](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

The notebook processes 14 disease labels and uses an 80/20 training-validation split. Images are resized to 224 × 224 and normalized, with random rotation and horizontal flipping applied during training.

## Results

Results from the saved notebook run after 3 epochs per model:

| Model | Validation Loss | Macro ROC-AUC | Label-wise Accuracy |
|---|---:|---:|---:|
| DenseNet-121 | 0.1442 | 0.8374 | 95.05% |
| EfficientNet-B0 | 0.1454 | 0.8289 | 95.06% |
| Hybrid CNN-Transformer | 0.1491 | 0.8190 | 94.88% |

*Accuracy refers to individual label decisions, not images with all 14 labels predicted correctly. These results are experimental and have not been clinically validated.*

## Explainability

Grad-CAM heatmaps are generated for all three models to visualize image regions influencing selected predictions. These visualizations are exploratory and do not establish medical correctness.

## How to Run

### Option 1: Kaggle (Recommended)

1. Open the [dataset on Kaggle](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized).
2. Create a Kaggle Notebook and attach the dataset.
3. Upload this repository's `.ipynb` notebook or open your existing notebook.
4. Enable GPU acceleration in notebook settings, if available.
5. Run the notebook cells in order.

The notebook uses Kaggle-specific dataset paths. If the dataset is mounted elsewhere, update `BASE_DIR`, `IMAGE_DIR`, and `CSV_PATH` accordingly.

### Option 2: Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   cd YOUR-REPOSITORY
   ```

2. Create and activate a virtual environment (Windows):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle, update the dataset paths in the notebook, and run the cells in order.

**Note:** Training all three models may take considerable time and requires sufficient memory and computing resources. Running the notebook from the beginning trains the models again.

## Limitations

- Each model was trained for only three epochs in the documented run.
- The split is image-level, not patient-level.
- Further evaluation using per-label precision, recall, F1-score, and independent test data would strengthen the results.

## License

This project is licensed under the MIT License. Dataset usage remains subject to the dataset's own terms.

## Disclaimer

For educational and research purposes only. This project is not a clinically validated diagnostic system and must not be used for medical decision-making.
