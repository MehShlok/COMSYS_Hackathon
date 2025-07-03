# Task B : 🚀 FaceNet Identity Verification

## ⚙️ Setup your environment

```bash
# clone your repo if needed
git clone https://github.com/MehShlok/COMSYS_Hackathon.git
cd face_net_implementation
```
### ✅ Using conda environment.yml
```bash
conda env create -f environment.yml
conda activate pytorch_gpu_environ
```

### ✅ Or manually install
```bash
conda create -n pytorch_gpu_environ python=3.9
conda activate pytorch_gpu_environ
pip install torch torchvision facenet-pytorch scikit-learn pillow opencv-python tqdm
```

---
## 🛠 Install your project (with setup.py)

```bash
pip install -e .
```

---

## 🔥 Fine-tune on your dataset

### ➡️ From scratch
```bash
python src/facenet_sim/finetune.py \
    --train_dir "Task_B/train" \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.0001 \
    --save_path "best_facenet_model.pth"
```

---

### ➡️ Continue fine-tuning from existing model
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

## 🚀 Run inference validation

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

## 📈 Custom parameters

### ➡️ Use a stricter threshold
```bash
--threshold 0.60
```

### ➡️ Change learning rate on fine-tune
```bash
--lr 0.00005
```

---