from abc import abstractstaticmethod

# import wsq  # needed for loading nist sd 14 dataset
import torch
import torchvision.transforms.functional as VTF
import cv2

from flx.setup.config import INPUT_SIZE
from flx.data.image_helpers import (
    pad_and_resize_to_deepprint_input_size,
)
from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.file_index import FileIndex


class ImageLoader(DataLoader):
    def __init__(self, root_dir: str):
        self._files: FileIndex = FileIndex(
            root_dir, self._extension(), self._file_to_id_fun
        )

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get(self, identifier: Identifier) -> torch.Tensor:
        return self._load_image(self._files.get(identifier))

    @abstractstaticmethod
    def _extension() -> str:
        pass

    @abstractstaticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        pass

    @abstractstaticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        pass


class SFingeLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".png"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/<subject_id>_<impression_id>.png
        subject_id, impression_id = filename.split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
        return VTF.to_tensor(img[:-32])


class FVC2004Loader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".tif"

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        # Pattern: <dir>/<subject_id>_<sample_id>.tif
        filename_without_ext = filename.replace(".tif", "")
        subject_id, impression_id = filename_without_ext.split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


class MCYTOpticalLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".bmp"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/<person>_<finger>_<impression>.png
        _, person, finger, impression = filename.split("_")
        # 12 impressions per finger
        subject = (10 * int(person)) + int(finger)
        return Identifier(subject, int(impression))

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        width = img.shape[1]
        return pad_and_resize_to_deepprint_input_size(img, roi=(310, width), fill=1.0)


class MCYTCapacitiveLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".bmp"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/<person:04d>_<finger>_<impression>.png
        _, person, finger, impression = filename.split("_")
        # 12 impressions per finger
        subject = (10 * int(person)) + int(finger)
        return Identifier(subject, int(impression))

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


class NistSD4Dataset(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".png"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/[f|s]<subject:04d>_<finger:02d>.png
        sample = 0 if filename[0] == "f" else 1
        subject, _ = filename[1:].split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(
            subject=int(subject) - 1,
            impression=sample,
        )

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


class SOCOFingLoader(ImageLoader):
    """
    Loader for SOCOFing dataset
    Filename pattern: <subject_id>__<gender>_<hand>_<finger_name>.BMP
    Example: 1__M_Right_little_finger.BMP

    Each subject has 10 fingerprints (5 fingers × 2 hands).
    We treat each finger as a separate "impression" for that subject.
    """

    @staticmethod
    def _extension() -> str:
        return ".BMP"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        parts = filename.replace(".BMP", "").split("__")
        person_id = int(parts[0])

        # Parse hand and finger info
        # Format: <gender>_<hand>_<finger_name>
        finger_info = parts[1]

        # Each finger is a SEPARATE SUBJECT
        # Create unique subject ID: person_id * 10 + finger_index
        # Left hand: 0-4, Right hand: 5-9
        if "Left" in finger_info:
            hand_offset = 0
        else:  # Right
            hand_offset = 5

        if "thumb" in finger_info:
            finger_idx = 0
        elif "index" in finger_info:
            finger_idx = 1
        elif "middle" in finger_info:
            finger_idx = 2
        elif "ring" in finger_info:
            finger_idx = 3
        elif "little" in finger_info:
            finger_idx = 4
        else:
            finger_idx = 0

        # Subject = unique finger (person_id * 10 + finger_index)
        # Person 1-600, each with 10 fingers (indices 0-9)
        subject_id = (person_id - 1) * 10 + hand_offset + finger_idx

        impression_id = 0

        return Identifier(subject=subject_id, impression=impression_id)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Convert to tensor first, then resize
        img_tensor = VTF.to_tensor(img)
        # Resize directly to target size
        return VTF.resize(img_tensor, (INPUT_SIZE, INPUT_SIZE), antialias=True)


class CrossmatchLoader(ImageLoader):
    """
    Loader for Crossmatch dataset with multiple impressions per finger.
    Filename pattern: xxx_yyy_zzz.tif
    where xxx = person ID, yyy = finger ID, zzz = scan number
    Example: 012_1_1.tif

    This dataset has multiple scans of the same finger,
    making it ideal for verification training.
    """

    @staticmethod
    def _extension() -> str:
        return ".tif"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        filename_without_ext = filename.replace(".tif", "")
        parts = filename_without_ext.split("_")
        person_id = int(parts[0])
        finger_id = int(parts[1])
        scan_number = int(parts[2])

        # Create unique subject ID: person_id * 10 + finger_id
        # This treats each finger as a separate subject for training
        subject_id = person_id * 10 + finger_id

        # scan_number becomes the impression_id
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(subject=subject_id - 1, impression=scan_number - 1)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_tensor = VTF.to_tensor(img)
        return VTF.resize(img_tensor, (INPUT_SIZE, INPUT_SIZE), antialias=True)
