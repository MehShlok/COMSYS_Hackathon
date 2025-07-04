# Task A : 🚀 Gender Classification
In this task we build a binary classification model to distinguish between male and female batches. The provided dataset was highly imbalanced, with a large disparity between the number of male and female face images. This posed a risk of model bias toward the majority class (male).To mitigate the class imbalance, we oversampled the minority class (female) by generating high-quality synthetic images using StyleGAN2-ADA.
StyleGAN2-ADA is a data-efficient GAN capable of generating realistic images with limited data. We fine-tuned the generator on the Flickr-Faces-HQ Dataset (FFHQ) dataset to synthesize new samples that preserved diversity and visual fidelity. The resulting dataset was balanced between male and female images, enabling the model to learn more equally from both classes.

We selected ResNet-18 as our classifier, a proven convolutional neural network with residual connections that allow deeper networks to train effectively. Pretrained on ImageNet, then fine-tuned on our face dataset. We benefitted from transfer learning as it gives better feature extraction from limited data. It is also lightweight and fast, making it suitable for experimentation and potential deployment.
Evaluated using accuracy, precision, recall, and F1-score on the validation set (real images only)

## 📊 Results :

### 🧪 Final Evaluation Metrics

| Metric        | Female      | Male        |
|---------------|-------------|-------------|
| Precision     | 85.71%      | 93.04%      |
| Recall        | 68.35%      | 97.38%      |
| F1 Score      | 76.06%      | 95.16%      |

- **Validation Accuracy:** `91.94%`
- **Training Loss:** `2.31e-07`

---

### 🧠 Observations

- The model achieved **high accuracy and strong precision for both classes**.
- **Recall for female faces is notably lower** (68%), indicating the model misses some true positives in that class.
- **Male class is well-learned**, with near-perfect recall (97%).
- The class imbalance was addressed using **StyleGAN2-ADA**, but some **domain shift between synthetic and real female images may still exist**.



# Task B : 🚀 FaceNet Identity Verification

## Setting up environment

```bash
# clone our repo if needed
git clone https://github.com/MehShlok/COMSYS_Hackathon.git
cd face_net_implementation
```
### Using conda environment.yml
```bash
conda env create -f environment.yml
conda activate pytorch_gpu_environ
```

### Or manually install
```bash
conda create -n pytorch_gpu_environ python=3.9
conda activate pytorch_gpu_environ
pip install torch torchvision facenet-pytorch scikit-learn pillow opencv-python tqdm
```

---
## Install our project (with setup.py)

```bash
pip install -e .
```

---

## Fine-tune on your dataset

### From scratch
```bash
python src/facenet_sim/finetune.py \
    --train_dir "Task_B/train" \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.0001 \
    --save_path "best_facenet_model.pth"
```

---

### Continue fine-tuning from existing model
```bash
python src/facenet_sim/finetune.py \
    --train_dir "Task_B/train" \
    --model_path "best_facenet_model.pth" \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.00005 \
    --save_path "best_facenet_model_v2.pth"
```

---

## Run inference validation

```bash
python src/facenet_sim/inference.py \
    --val_dir "Task_B/val" \
    --model_path "best_facenet_model.pth" \
    --threshold 0.55
```

It will print:
```
Validation Accuracy: 2345/2954 = 79.38%
```

---

## Custom parameters

### Use a stricter threshold
```bash
--threshold 0.60
```

### ➡️ Change learning rate on fine-tune
```bash
--lr 0.00005
```

---


# Model Weights

This repository contains the code for our project.  
The trained model weights are stored separately on Google Drive due to their size.

## 📂 Download Model Weights

You can download the model weights from the following Google Drive link:

[**Download Weights from Google Drive**](https://drive.google.com/drive/folders/1nMuMpQECgC6BL7MJKBsZoT2lPy024xQ_?usp=sharing)

## 🔄 How to Use

1. Click the link above to open the Google Drive folder.
2. Download all the files inside the folder to your local machine.
3. Place the downloaded weights in the `models/` directory (or wherever your code expects the weights).
4. Ensure your code correctly loads the weights. 

