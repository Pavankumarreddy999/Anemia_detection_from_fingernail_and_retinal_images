# Anemia Detection via Deep Learning

We are developing parallel deep learning pipelines for non-invasive anemia detection across multiple image modalities, including fingernail, palpebral conjunctiva, and retinal images. This repository currently contains the **Fingernail Image Modality** implementation. 

Our fingernail model (EfficientNetB0) is designed to achieve strong baseline accuracy by leveraging advanced image preprocessing, data augmentation, and a robust two-phase training strategy.

## Key Features

- **Intelligent Data Parsing**: Automatically parses filenames to correctly label images as `Anemic` or `Non-Anemic`, robustly handling spelling mistakes and various file naming conventions.
- **Advanced Preprocessing (CLAHE)**: Employs Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Lightness (L) channel of the LAB color space. This acts as a "doctor's magnifying glass" to enhance the color contrast of the nail bed, isolating brightness from color to make hemoglobin levels more distinguishable for the model.
- **Robust Data Augmentation**: Utilizes extensive on-the-fly data augmentation (rotation, shifting, zooming, and flipping) to prevent overfitting and improve model generalization.
- **Class Imbalance Handling**: Automatically computes and applies balanced class weights during training to ensure the model isn't biased towards the majority class.

## Architecture & Training Pipeline

The fingernail pipeline relies on **EfficientNetB0** architecture trained via a two-phase transfer learning approach:

1. **Phase 1: Feature Extraction (Top Layer Training)**
   - The EfficientNetB0 base model is frozen.
   - A custom classification head is trained (GlobalAveragePooling2D → BatchNormalization → Dense (256) → Dropout (0.5) → Dense (1, Sigmoid)).
   - Optimized using Adam (lr=0.001) with `EarlyStopping` and `ReduceLROnPlateau`.
2. **Phase 2: Advanced Fine-Tuning**
   - The base model is unfrozen (except for BatchNormalization layers to preserve moving averages and ensure stability).
   - The entire network is fine-tuned using a lower learning rate (Adam, lr=1e-5) and `BinaryCrossentropy` with label smoothing (0.1) for better calibration and generalization.

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV (`cv2`)
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

## Usage

1. Open the Jupyter Notebook: `anemia-detection-fingernail (1).ipynb`
2. Ensure you have the dataset available (the notebook is configured to automatically find the Kaggle input path, e.g., `/kaggle/input/...`).
3. Run the notebook cells sequentially to:
   - Load and split the dataset (80% Train / 20% Val)
   - Apply CLAHE image enhancement
   - Train the two-phase model
   - View accuracy/loss charts and detailed classification reports
4. The trained model will be exported as `anemia_detector_efficientnet.keras`.

## Future Work

- Expanding the repository to include the conjunctiva and retinal pipelines.
- Incorporating Grad-CAM explainability across all modalities to highlight hemoglobin-relevant features, making the predictions clinically interpretable.
