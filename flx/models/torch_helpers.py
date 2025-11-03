import os
import time

import torch


CUDA_DEVICE = 0
TRAIN_ON_A_100 = False  # Set to False for RTX 2080 (8GB) - True only for A100 (40GB)

# MPS (Apple Silicon) limitations:
# - MPS may not support all operations with float64 (double precision)
# - Some operations may fall back to CPU automatically
# - num_workers should be 0 for DataLoader on MPS
MPS_FALLBACK_TO_CPU_ON_ERROR = True


def get_dataloader_args(train: bool) -> dict:
    batch_size = 8  # Reduced from 16 for RTX 2080 (8GB)
    if not train:
        batch_size *= 2  # More memory available without gradients

    if torch.cuda.is_available():
        if TRAIN_ON_A_100:  # Use 40GB graphics ram by preloading to pinned memory
            return {
                "batch_size": batch_size,
                "shuffle": train,
                "num_workers": 16,
                "prefetch_factor": 2,
                "pin_memory": True,
                "pin_memory_device": f"cuda:{CUDA_DEVICE}",
            }
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 6,
            "prefetch_factor": 3,
            "pin_memory": True,
            "pin_memory_device": f"cuda:{CUDA_DEVICE}",
            "persistent_workers": True,
        }
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) configuration
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 0,  # MPS works best with num_workers=0
            "prefetch_factor": None,  # Not applicable when num_workers=0
        }
    else:
        # CPU configuration
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 4,
            "prefetch_factor": 1,
        }


def get_device() -> str:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{CUDA_DEVICE}")
    elif torch.backends.mps.is_available():
        # Test if MPS can handle basic operations
        try:
            # Test basic tensor operations on MPS
            test_tensor = torch.randn(2, 2, device="mps")
            _ = test_tensor * 2.0
            return torch.device("mps")
        except Exception as e:
            print(f"Warning: MPS available but failed basic test: {e}")
            if MPS_FALLBACK_TO_CPU_ON_ERROR:
                print("Falling back to CPU")
                return torch.device("cpu")
            else:
                return torch.device("mps")  # Let user handle MPS issues
    return torch.device("cpu")


def save_model_parameters(
    full_param_path: str,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    """
    Tries to save the parameters of model and optimizer in the given path
    """
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
            },
            full_param_path,
        )
    except KeyboardInterrupt:
        print("\n>>>>>>>>> Model is being saved! Will exit when done <<<<<<<<<<\n")
        save_model_parameters(full_param_path, model, optim)
        time.sleep(10)
        raise KeyboardInterrupt()


def load_model_parameters(
    full_param_path: str,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    """
    Tries to load the parameters stored in the given path
    into the given model and optimizer.
    """
    if not os.path.exists(full_param_path):
        raise FileNotFoundError(f"Model file {full_param_path} did not exist.")
    checkpoint = torch.load(full_param_path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    if loss is not None:
        loss.load_state_dict(checkpoint["loss_state_dict"])
    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
