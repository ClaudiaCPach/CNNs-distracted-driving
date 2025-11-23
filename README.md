# CNNs-distracted-driving
Distracted Driver Classification (AUC Dataset V2, 10 classes, 44 drivers). Baseline CNNs, MediaPipe crops, attention/fusion, cross-camera generalization.

Use Python 3.11

## Dataset Setup

### Local Development (Mac/PC)
1. Download the AUC Distracted Driver dataset
2. Create the following folder structure in your home directory:
   ```
   ~/TFM/
   ├── data/
   │   └── auc.distracted.driver.dataset_v2/
   │       ├── v1_cam1_no_split/
   │       └── v2_cam1_cam2_split_by_driver/
   ├── outputs/
   └── checkpoints/
   ```
3. Extract the dataset into `~/TFM/data/auc.distracted.driver.dataset_v2/`

### Google Colab
- The Colab setup notebook will handle mounting Google Drive and setting up paths automatically