# Multi-Label Chest Disease Classification

A deep learning project that predicts multiple chest disease labels from chest X-ray images using PyTorch and computer vision. This project compares three deep learning architectures and uses Grad-CAM to visualize image regions that contribute to model predictions.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used for medical decision-making.

## 📌 Project Overview

Chest X-rays can contain multiple abnormalities simultaneously. This project uses a **multi-label classification** approach to predict the presence of 14 different diseases or findings from a single X-ray image.

The project explores transfer learning, model evaluation, and visual explainability using the following architectures:

- DenseNet-121
- EfficientNet-B0
- Hybrid CNN-Transformer (CvT-style)

Grad-CAM visualizations are also generated to help interpret which regions of an X-ray influence selected model predictions.

## 🚀 Features

- Multi-label classification of 14 chest diseases.
- Transfer learning using pretrained ImageNet weights.
- Comparison of three deep learning architectures.
- Image preprocessing and data augmentation.
- Evaluation using validation loss, macro ROC-AUC, and label-wise accuracy.
- Grad-CAM heatmaps for model explainability.
- Single-image inference and visualization.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision:** OpenCV, Pillow
- **Data Processing:** Pandas, NumPy
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Kaggle Notebooks with GPU support

## 📂 Dataset

This project uses the NIH Chest X-ray 14 dataset with resized 224 × 224 images.

**Dataset:** [NIH Chest X-ray 14 — 224×224 Resized](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

The notebook loads 112,120 metadata records and uses the `Finding Labels` column to create binary targets for 14 disease labels.

### Disease Labels

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

Each image can have multiple positive labels, making this a multi-label classification problem rather than a conventional single-class classification task.

## 🧠 Models Implemented

### 1. DenseNet-121

- Uses pretrained ImageNet weights.
- Replaces the final classification layer with a 14-output layer.
- Trained for multi-label chest disease prediction.

### 2. EfficientNet-B0

- Uses pretrained ImageNet weights.
- Replaces the final classifier with a 14-output layer.
- Evaluated alongside DenseNet-121.

### 3. Hybrid CNN-Transformer

- Uses a pretrained ResNet-34 convolutional backbone.
- Converts the extracted spatial feature map into a sequence of 49 feature tokens.
- Processes these tokens using a Transformer encoder with 2 layers and 8 attention heads.
- Uses a final linear layer to predict the 14 labels.

## ⚙️ Data Preprocessing

The notebook applies the following preprocessing steps:

- Resizes images to 224 × 224 pixels.
- Converts images into PyTorch tensors.
- Normalizes images using ImageNet mean and standard deviation.
- Applies random rotation up to 10 degrees during training.
- Applies random horizontal flipping during training.

Random augmentation is applied to training images, while validation images use resizing, tensor conversion, and normalization without random augmentation.

### Train-Validation Split

The dataset is divided into:

- **80% Training**
- **20% Validation**

The split uses `train_test_split` with `random_state=42`.

**Note:** The notebook uses a random image-level split rather than a patient-wise split. This is an important limitation when interpreting the validation results.

## 🔬 Training Configuration

| Parameter | Configuration |
|---|---|
| Deep learning framework | PyTorch |
| Input image size | 224 × 224 |
| Number of labels | 14 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | BCEWithLogitsLoss |
| Epochs per model | 3 |
| Validation metrics | Macro ROC-AUC and label-wise accuracy |
| Checkpoint selection | Lowest validation loss |

The notebook uses sigmoid probabilities to evaluate individual label predictions and calculates macro ROC-AUC across the disease labels.

## 📊 Results

The following results are from the saved notebook run after three training epochs per model.

| Model | Validation Loss | Macro ROC-AUC | Label-wise Accuracy |
|---|---:|---:|---:|
| DenseNet-121 | 0.1442 | 0.8374 | 95.05% |
| EfficientNet-B0 | 0.1454 | 0.8289 | 95.06% |
| Hybrid CNN-Transformer | 0.1491 | 0.8190 | 94.88% |

### Results Interpretation

- **DenseNet-121** achieved the highest macro ROC-AUC of 0.8374 in this run.
- **EfficientNet-B0** achieved the highest reported label-wise accuracy of 95.06%.
- **Hybrid CNN-Transformer** provided an alternative architecture combining convolutional feature extraction and Transformer-based processing.

The reported accuracy measures correct individual label decisions across the dataset. It does not mean that 95% of X-ray images had all 14 labels predicted correctly.

These results represent a limited experimental run and should not be interpreted as evidence of clinical diagnostic performance.

## 🔍 Model Explainability with Grad-CAM

Grad-CAM is used to visualize image regions that contribute to selected model predictions.

The notebook implements Grad-CAM for DenseNet-121, EfficientNet-B0, and the Hybrid CNN-Transformer model. The generated heatmaps are overlaid on the original chest X-rays using OpenCV.

For the example image shown in the notebook, the top predicted label was **Infiltration** across all three models.

| Model | Example Top Label | Reported Score |
|---|---|---:|
| DenseNet-121 | Infiltration | 10.80% |
| EfficientNet-B0 | Infiltration | 11.28% |
| Hybrid CNN-Transformer | Infiltration | 9.79% |

These scores are model predictions for a single example image, not confirmed diagnoses. Grad-CAM is an exploratory visualization technique and does not establish clinical correctness.

## 📁 Repository Structure

```text
Multi-Label-Chest-Disease-Classification/
│
├── README.md
└── Multi-Label Chest Disease Classification.ipynb
```

The trained model checkpoints are saved in the Kaggle working directory during the notebook run:

```text
best_densenet121.pth
best_efficientnet.pth
best_cvt.pth
```

These checkpoint files are not automatically included when uploading the notebook to GitHub.

## 💻 Running the Project

### Option 1: Run on Kaggle

1. Open the [Kaggle dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized).
2. Create a Kaggle Notebook or open your existing notebook.
3. Attach the dataset to the notebook.
4. Enable GPU acceleration in the notebook settings, if available.
5. Run the notebook cells in order.

The notebook uses Kaggle-specific dataset paths:

```python
BASE_DIR = "/kaggle/input/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized"

IMAGE_DIR = f"{BASE_DIR}/images-224/images-224"

CSV_PATH = f"{BASE_DIR}/Data_Entry_2017.csv"
```

If the dataset is mounted at a different location, update these paths accordingly.

### Option 2: Install Dependencies Locally

Install the main Python dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn pillow matplotlib opencv-python tqdm
```

Download the dataset from Kaggle and update the dataset paths in the notebook before running it locally.

**Note:** Training all three models again can require significant computation time. Viewing the notebook on GitHub does not execute its cells.

## ⚠️ Limitations

- Each model was trained for only three epochs in the documented run.
- The validation split is image-level rather than patient-level.
- Class imbalance may affect model performance.
- Accuracy alone is insufficient to assess multi-label medical classification.
- Additional per-label precision, recall, F1-score, and threshold analysis would strengthen the evaluation.
- The models and Grad-CAM visualizations have not been clinically validated.
- Results may vary depending on the dataset split, hardware, and training configuration.

## 🔮 Future Improvements

- Train models for more epochs and tune hyperparameters.
- Evaluate using patient-wise data splits.
- Report per-disease precision, recall, and F1-score.
- Tune decision thresholds for individual labels.
- Compare model performance on an independent dataset.
- Improve experiment tracking and model checkpoint management.
- Further investigate model explainability and prediction reliability.

## 📜 License

This project is licensed under the **MIT License**.

The MIT License applies to the project code and does not automatically grant permission to redistribute the dataset or third-party materials. Refer to the original dataset terms before redistributing any data.

## ⚕️ Disclaimer

This project is intended solely for educational and research purposes. It is not a substitute for professional medical evaluation and must not be used for diagnosis, screening, treatment, or clinical decision-making.
