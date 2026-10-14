# Multi-Label Chest Disease Classification

A deep learning project that predicts multiple chest-disease labels from chest X-ray images. The notebook trains and compares three PyTorch architectures and uses Grad-CAM to visualize image regions that contribute to selected predictions.

> **Research/educational use only.** This project is not a clinically validated diagnostic system and must not be used to make medical decisions.

## Project Overview

Chest X-rays may show more than one finding at a time. This project treats the task as **multi-label classification**: the model produces a score for each of 14 labels rather than choosing only one class.

The notebook uses transfer learning with pretrained ImageNet weights and compares:

- **DenseNet-121**
- **EfficientNet-B0**
- **Hybrid CNN-Transformer (CvT-style)** — a pretrained ResNet-34 feature extractor followed by a Transformer encoder

It also includes single-image inference and Grad-CAM visualizations for all three models.

## Dataset

The notebook is configured for the Kaggle dataset:

**NIH Chest X-ray 14 — 224×224 resized images**  
[Kaggle dataset page](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

- **Records loaded:** 112,120
- **Labels:** 14 disease/finding labels
- **Image size used by the notebook:** 224 × 224
- **Split:** 80% training / 20% validation using `train_test_split(test_size=0.2, random_state=42)`

The labels are derived from the dataset's `Finding Labels` metadata column. The notebook creates one binary target column for each label.

### Labels

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

> The notebook uses a random image-level split. It does not implement a patient-wise split, so this should be considered when interpreting validation results.

## Models and Method

### 1. DenseNet-121
Loads pretrained ImageNet DenseNet-121 weights and replaces the final classifier with a 14-output linear layer.

### 2. EfficientNet-B0
Loads pretrained ImageNet EfficientNet-B0 weights and replaces its final classifier with a 14-output linear layer.

### 3. Hybrid CNN-Transformer
Uses a pretrained ResNet-34 convolutional backbone. Its final spatial feature map is reshaped into a sequence of 49 feature tokens, processed by a Transformer encoder with 2 layers and 8 attention heads, then mean-pooled and passed to a 14-output classifier.

### Training configuration

| Setting | Value |
|---|---|
| Framework | PyTorch / torchvision |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Batch size | 32 |
| Epochs per model | 3 |
| Input size | 224 × 224 |
| Compute device in saved run | CUDA GPU |
| Checkpoint selection | Lowest validation loss |

Training transforms resize images, apply random rotation up to 10 degrees and random horizontal flipping, convert images to tensors, and normalize with ImageNet mean and standard deviation. Validation transforms use resizing, tensor conversion, and the same normalization without random augmentation.

## Results

The table below reports the metrics saved in the notebook after **3 epochs per model**. Values are from this particular run and are not a guarantee of performance on other data.

| Model | Final validation loss | Final validation macro ROC-AUC | Reported label-wise validation accuracy |
|---|---:|---:|---:|
| DenseNet-121 | 0.1442 | 0.8374 | 95.05% |
| EfficientNet-B0 | 0.1454 | 0.8289 | 95.06% |
| Hybrid CNN-Transformer | 0.1491 | 0.8190 | 94.88% |

**Metric note:** The notebook's reported “accuracy” is calculated by applying sigmoid to each output, thresholding each label at 0.5, and counting correct individual label decisions across all labels. It is **not exact-match accuracy across all 14 labels for an image**. For imbalanced multi-label medical datasets, accuracy should be interpreted alongside metrics such as per-label precision, recall, F1-score, and ROC-AUC.

The best final macro ROC-AUC in this run was **0.8374 with DenseNet-121**. EfficientNet-B0 had the highest reported label-wise accuracy at **95.06%**. All models were trained for only three epochs in the saved notebook; further experimentation and stronger evaluation would be needed before drawing broad conclusions.

## Explainability with Grad-CAM

The notebook implements model-specific Grad-CAM helpers for DenseNet-121, EfficientNet-B0, and the hybrid model. It generates a heatmap for a selected output label and overlays it on the original X-ray using OpenCV.

The notebook's example image produced these top scores:

| Model | Top predicted label | Score shown in notebook |
|---|---|---:|
| DenseNet-121 | Infiltration | 10.80% |
| EfficientNet-B0 | Infiltration | 11.28% |
| Hybrid CNN-Transformer | Infiltration | 9.79% |

These are model output scores for one example image, **not verified diagnoses**. Grad-CAM visualizations are exploratory explanations and do not establish that a model is clinically correct or that a highlighted region is medically meaningful.

## Repository Contents

A suggested repository structure:

```text
.
├── README.md
└── Multi-Label Chest Disease Classification.ipynb
```

The notebook saves model checkpoints in Kaggle's working directory:

- `best_densenet121.pth`
- `best_efficientnet.pth`
- `best_cvt.pth`

These checkpoint files are **not automatically included** when you upload the notebook to GitHub. If you want others to run inference without retraining, decide how to distribute the weights separately and check the dataset/model terms and GitHub file-size limits first.

## Running the Notebook

1. Open the Kaggle dataset linked above and add it to your Kaggle notebook.
2. Open the notebook in Kaggle.
3. Enable a GPU accelerator in notebook settings if available.
4. Run the cells in order.

The notebook currently uses Kaggle-specific paths:

```python
BASE_DIR = "/kaggle/input/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized"
IMAGE_DIR = f"{BASE_DIR}/images-224/images-224"
CSV_PATH = f"{BASE_DIR}/Data_Entry_2017.csv"
```

If running outside Kaggle, update these paths to match your local dataset layout.

### Main dependencies

The notebook imports:

- Python
- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- Pillow
- Matplotlib
- OpenCV (`opencv-python`)
- tqdm

A simple starting install command is:

```bash
pip install torch torchvision pandas numpy scikit-learn pillow matplotlib opencv-python tqdm
```

Install a PyTorch build compatible with your hardware if you plan to train on a GPU. Training the models again can take considerable time; viewing the notebook on GitHub does not execute its cells.

## Limitations and Future Work

- Each architecture was trained for only three epochs in this run.
- The notebook uses a random image-level split rather than a patient-wise split.
- The saved evaluation emphasizes macro ROC-AUC and label-wise threshold accuracy; additional per-label precision, recall, F1, calibration, and confusion analysis would improve the evaluation.
- The model scores and Grad-CAM maps are not clinically validated.
- Results may vary with random initialization, hardware, library versions, and training configuration.

Possible next steps include longer training with a documented validation strategy, patient-wise evaluation, threshold tuning on validation data, per-label metric reporting, and testing on an independent dataset.

## License

This repository is released under the **MIT License**. The license for this code does not automatically grant rights to redistribute the chest X-ray dataset or other third-party materials. Please follow the dataset's own terms.

## Disclaimer

This project is intended for learning and research purposes only. It is not intended for diagnosis, screening, treatment, or any other clinical use.
