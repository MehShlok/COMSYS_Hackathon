import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = []
        self.people_dirs = [os.path.join(root_dir, person) 
                            for person in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, person))]

        for person_dir in self.people_dirs:
            distortion_dir = os.path.join(person_dir, "distortion")
            if not os.path.isdir(distortion_dir):
                continue

            clean_images = [f for f in os.listdir(person_dir)
                            if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('._')]

            for clean_img in clean_images:
                clean_name = os.path.splitext(clean_img)[0]
                distorted_versions = [f for f in os.listdir(distortion_dir) 
                                      if f.startswith(clean_name + "_")]

                if len(distorted_versions) == 0:
                    continue

                self.triplets.append({
                    "anchor": os.path.join(person_dir, clean_img),
                    "positives": [os.path.join(distortion_dir, d) for d in distorted_versions],
                    "person_dir": person_dir
                })

        if len(self.triplets) == 0:
            raise RuntimeError("No valid triplets found. Check your folder structure.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor_path = triplet["anchor"]
        positive_path = random.choice(triplet["positives"])

        while True:
            neg_person_dir = random.choice(self.people_dirs)
            if neg_person_dir != triplet["person_dir"]:
                break

        neg_clean_imgs = [f for f in os.listdir(neg_person_dir)
                          if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('._')]
        neg_img_path = os.path.join(neg_person_dir, random.choice(neg_clean_imgs))

        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(neg_img_path)

        return anchor, positive, negative

    def _load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def load_facenet(model_path, device):
    if model_path is None:
        print("Loading default pretrained FaceNet (facenet-pytorch InceptionResnetV1)")
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    else:
        print(f"Loading fine-tuned FaceNet from {model_path}")
        model = torch.load(model_path, map_location=device)
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Training data directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to existing facenet model .pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="finetuned_facenet.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    facenet = load_facenet(args.model_path, device)
    facenet.train()

    transform = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = TripletFaceDataset(root_dir=args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(facenet.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        facenet.train()
        epoch_loss = 0
        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embed = facenet(anchor)
            pos_embed = facenet(positive)
            neg_embed = facenet(negative)

            loss = triplet_loss(anchor_embed, pos_embed, neg_embed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(facenet, args.save_path)
    print(f"Model saved to {args.save_path}")
