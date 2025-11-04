import base64
from io import BytesIO
from typing import Union
import numpy as np
import cv2
import torch
from PIL import Image

from flx.extractor.fixed_length_extractor import (
    get_DeepPrint_TexMinu,
    DeepPrintExtractor,
)

from flx.data.dataset import (
    Dataset,
    Identifier,
    IdentifierSet,
    ConstantDataLoader,
)

from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size


class Generator:

    def __init__(
        self,
        model_path: str = "trained_models",
        num_subjects: int = 8000,
        embedding_dims: int = 256,
        use_binarization: bool = True,
        ridge_width: float = 5.0,
    ):
        self.model_path = model_path
        self.num_subjects = num_subjects
        self.embedding_dims = embedding_dims
        self.use_binarization = use_binarization
        self.ridge_width = ridge_width

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print("Loading Model Instance: ")
        print(f"   Device: {self.device}")
        print(f"   Model path: {model_path}")
        print(f"   Architecture: TexMinu ({embedding_dims}+{embedding_dims})")
        print(f"   Training subjects: {num_subjects}")

        self._load_model()

        self.binarizer = (
            LazilyAllocatedBinarizer(ridge_width) if use_binarization else None
        )

    def _load_model(self):
        try:
            self.extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
                num_training_subjects=self.num_subjects,
                num_dims=self.embedding_dims,
            )

            self.extractor.load_best_model(self.model_path)
            self.extractor.model.to(self.device)
            self.extractor.model.eval()

            print("Model loaded successfully ")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        try:
            preprocessed = pad_and_resize_to_deepprint_input_size(image, fill=1.0)

            if self.binarizer is not None:
                preprocessed = self.binarizer(preprocessed)

            if len(preprocessed.shape) == 3:
                preprocessed = preprocessed.unsqueeze(0)
            elif len(preprocessed.shape) == 2:
                preprocessed = preprocessed.unsqueeze(0).unsqueeze(0)
            preprocessed = preprocessed.to(self.device)

            return preprocessed

        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")

    def _load_image(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """
        Convert various image input types to numpy array.
        """
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, str):
            if image.startswith("data:"):
                base64_data = image.split(",", 1)[1] if "," in image else image
                return self._decode_base64_image(base64_data)
            else:
                return self._decode_base64_image(image)

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 string to grayscale numpy array."""
        try:
            img_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to decode image data")
            return img
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

    def _base64_to_array(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy array."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)
        return self._bytes_to_array(image_bytes)

    def _bytes_to_array(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "L":
            image = image.convert("L")
        return np.array(image)

    def extract_embedding(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # Preprocess the image to get tensor
        preprocessed = self.preprocess_image(image)

        # Create a minimal dataset with single image (matches notebook approach)
        # Use a dummy identifier
        dummy_id = Identifier(subject=0, impression=0)

        # Create a constant data loader that returns the preprocessed tensor
        # Remove batch dimension since Dataset expects (C, H, W) not (1, C, H, W)
        tensor_without_batch = preprocessed.squeeze(0)
        loader = ConstantDataLoader(tensor_without_batch)

        # Create dataset with single image
        id_set = IdentifierSet([dummy_id])
        dataset = Dataset(loader, id_set)

        # Extract using the same method as notebook
        texture_embeddings, minutiae_embeddings = self.extractor.extract(dataset)

        # Get the embeddings for our single image
        texture_emb = texture_embeddings.get(dummy_id)
        minutiae_emb = minutiae_embeddings.get(dummy_id)

        return texture_emb, minutiae_emb

    def compare(
        self,
        image1: Union[np.ndarray, str, bytes],
        image2: Union[np.ndarray, str, bytes],
    ) -> float:
        """
        Compare two fingerprint images and return similarity score.
        Returns:
            Similarity score where:
        0.0 - 0.5: Very different fingerprints (definitely not a match)
        0.5 - 1.0: Different fingerprints (impostor comparisons, high confidence)
        1.0 - 1.5: Borderline (might be same finger with poor quality)
        1.5 - 2.0: Same fingerprint (genuine comparisons)
        ~2.0: Perfect match (same image compared to itself)
        """
        # Convert inputs to numpy arrays
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)

        tex1, min1 = self.extract_embedding(img1)
        tex2, min2 = self.extract_embedding(img2)

        emb1 = np.concatenate([tex1, min1])
        emb2 = np.concatenate([tex2, min2])

        similarity = np.dot(emb1, emb2)

        return float(similarity)
