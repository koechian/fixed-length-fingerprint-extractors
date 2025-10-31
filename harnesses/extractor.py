import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import time


class FingerprintFeatureExtractor:
    """
    Simple ResNet18-based feature extractor
    Extracts 512-dimensional global features from fingerprint images
    """

    def __init__(self, device=None, normalize=True):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            normalize: Whether to L2-normalize the output vectors
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.normalize = normalize

        # Load pre-trained ResNet18
        print("Loading ResNet18 (pre-trained on ImageNet)...")
        resnet = models.resnet18(pretrained=True)

        # Remove the final classification layer (fc)
        # This gives us the 512-D feature vector from avgpool layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Image preprocessing pipeline
        # Same as ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # ResNet expects 224x224
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert grayscale to 3-channel
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225],  # ImageNet std
                ),
            ]
        )

        print("Feature extractor ready!\n")

    def load_image(self, image_path):
        """
        Load and preprocess an image

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor ready for model
        """
        img = Image.open(image_path)

        # Convert to grayscale if not already
        if img.mode != "L":
            img = img.convert("L")

        # Apply transformations
        img_tensor = self.transform(img)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def extract(self, image_path):
        """
        Extract feature vector from a fingerprint image

        Args:
            image_path: Path to fingerprint image

        Returns:
            512-dimensional numpy array (float32)
        """
        # Load and preprocess
        img_tensor = self.load_image(image_path).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)

        # Convert to numpy: (1, 512, 1, 1) -> (512,)
        features = features.squeeze().cpu().numpy().astype(np.float32)

        # Normalize to unit length (for cosine similarity)
        if self.normalize:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

        return features

    def extract_batch(self, image_paths):
        """
        Extract features from multiple images in batch (faster)

        Args:
            image_paths: List of image paths

        Returns:
            Array of shape (N, 512)
        """
        # Load all images
        tensors = [self.load_image(path) for path in image_paths]
        batch = torch.cat(tensors, dim=0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(batch)

        # Convert to numpy: (N, 512, 1, 1) -> (N, 512)
        features = features.squeeze().cpu().numpy().astype(np.float32)

        # Normalize each vector
        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norms + 1e-8)

        return features

    def compare(self, embedding1, embedding2):
        """
        Compare two embeddings using cosine similarity

        Args:
            embedding1, embedding2: 512-D numpy arrays

        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        # Cosine similarity (dot product if normalized)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors

    Returns:
        Similarity score between -1 and 1 (higher = more similar)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-8)


def euclidean_distance(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors

    Returns:
        Distance (lower = more similar)
    """
    return np.linalg.norm(vec1 - vec2)


def main():
    parser = argparse.ArgumentParser(description="Extract fingerprint features")
    parser.add_argument("image", type=str, help="Path to fingerprint image")
    parser.add_argument("--compare", type=str, help="Optional: second image to compare")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--save", type=str, help="Save embedding to .npy file")
    args = parser.parse_args()

    # Initialize extractor
    extractor = FingerprintFeatureExtractor(device=args.device)

    # Extract features
    print(f"Extracting features from: {args.image}")
    start_time = time.time()
    embedding = extractor.extract(args.image)
    elapsed = time.time() - start_time

    print(f"✓ Extracted in {elapsed*1000:.2f}ms")
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"\nFirst 10 values: {embedding[:10]}")

    # Save if requested
    if args.save:
        np.save(args.save, embedding)
        print(f"\n✓ Saved embedding to: {args.save}")

    # Compare if second image provided
    if args.compare:
        print(f"\nComparing with: {args.compare}")
        embedding2 = extractor.extract(args.compare)

        similarity = cosine_similarity(embedding, embedding2)
        distance = euclidean_distance(embedding, embedding2)

        print(f"\n{'='*50}")
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Euclidean Distance: {distance:.4f}")
        print(f"{'='*50}")

        if similarity > 0.8:
            print("✓ HIGH similarity - likely same finger")
        elif similarity > 0.5:
            print("~ MEDIUM similarity - possibly related")
        else:
            print("✗ LOW similarity - likely different fingers")


if __name__ == "__main__":
    main()
