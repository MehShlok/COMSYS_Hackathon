import torch
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
import torch.serialization

# Needed for PyTorch >=2.6 to allow loading full model
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
torch.serialization.add_safe_globals([InceptionResnetV1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------
# Compute embedding
# ----------------------
def compute_embedding(img_path, model, transform):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img)
    return embedding

# ----------------------
# Validation loop
# ----------------------
def validate(val_dir, model, transform, threshold=0.55):
    correct, total = 0, 0
    person_list = [p for p in os.listdir(val_dir) if not p.startswith('._')]

    for person in tqdm(person_list, desc="Validating identities"):
        person_dir = os.path.join(val_dir, person)
        distortion_dir = os.path.join(person_dir, 'distortion')

        clean_imgs = [f for f in os.listdir(person_dir)
                      if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('._')]

        for clean_img in clean_imgs:
            clean_path = os.path.join(person_dir, clean_img)
            clean_name = os.path.splitext(clean_img)[0]
            clean_embedding = compute_embedding(clean_path, model, transform)

            distorted_versions = [f for f in os.listdir(distortion_dir)
                                  if f.startswith(clean_name + "_")]

            for distorted_file in distorted_versions:
                distorted_path = os.path.join(distortion_dir, distorted_file)
                distorted_embedding = compute_embedding(distorted_path, model, transform)

                sim = cosine_similarity(clean_embedding.cpu(), distorted_embedding.cpu())[0][0]
                if sim >= threshold:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nValidation Accuracy: {correct}/{total} = {accuracy:.2%}")

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth or state_dict")
    parser.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold")
    args = parser.parse_args()

    # ----------------------
    # Load full torch model object with safe globals
    # ----------------------
    try:
        facenet = torch.load(args.model_path, map_location=device, weights_only=False)
        print("✅ Loaded full torch model object.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)

    facenet.eval().to(device)

    # ----------------------
    # Transform
    # ----------------------
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # ----------------------
    # Run validation
    # ----------------------
    validate(args.val_dir, facenet, transform, threshold=args.threshold)
