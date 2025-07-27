"""
Device detection and management utilities for M1/Apple Silicon optimization.
Provides intelligent device selection with preference for MPS > CUDA > CPU.
"""

import torch
import warnings


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Prioritizes devices in the following order:
    1. MPS (Metal Performance Shaders) for Apple Silicon (M1/M2/M3)
    2. CUDA for NVIDIA GPUs
    3. CPU as fallback
    
    Returns:
        torch.device: The best available device for computation
    """
    # Check for Apple Silicon MPS support
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple Silicon (MPS) device: {device}")
        return device
    
    # Check for CUDA support
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device} ({cuda_device_name})")
        return device
    
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
        return device


def get_device_info():
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Dictionary containing device information
    """
    info = {
        'mps_available': False,
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_device_names': [],
        'selected_device': None
    }
    
    # Check MPS availability
    if hasattr(torch.backends, 'mps'):
        info['mps_available'] = torch.backends.mps.is_available()
    
    # Check CUDA availability
    info['cuda_available'] = torch.cuda.is_available()
    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(info['cuda_device_count'])]
    
    # Get selected device
    info['selected_device'] = get_device()
    
    return info


def move_to_device(obj, device):
    """
    Move tensor, model, or data structure to specified device.
    
    Args:
        obj: Object to move (tensor, model, list, dict, etc.)
        device: Target device
    
    Returns:
        Object moved to the specified device
    """
    if hasattr(obj, 'to'):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        return obj


def setup_model_for_device(model, device=None):
    """
    Setup model for optimal performance on the specified device.
    
    Args:
        model: PyTorch model to setup
        device: Target device (if None, uses get_device())
    
    Returns:
        model: Model moved to device and optimized
    """
    if device is None:
        device = get_device()
    
    # Move model to device
    model = model.to(device)
    
    # Enable optimizations for Apple Silicon
    if device.type == 'mps':
        # MPS-specific optimizations
        # Set memory format for better performance on Apple Silicon
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.backends.mps, 'enable_fallback'):
                torch.backends.mps.enable_fallback()
        except:
            pass
    
    # Enable optimizations for CUDA
    elif device.type == 'cuda':
        # CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    return model


def get_memory_usage(device=None):
    """
    Get memory usage information for the specified device.
    
    Args:
        device: Device to check (if None, uses current device)
    
    Returns:
        dict: Memory usage information
    """
    if device is None:
        device = get_device()
    
    memory_info = {'device': str(device)}
    
    if device.type == 'cuda':
        memory_info.update({
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device),
            'max_allocated': torch.cuda.max_memory_allocated(device),
            'max_reserved': torch.cuda.max_memory_reserved(device)
        })
    elif device.type == 'mps':
        # MPS memory monitoring (limited in current PyTorch versions)
        try:
            memory_info['allocated'] = torch.mps.current_allocated_memory()
        except:
            memory_info['allocated'] = 'Not available'
    else:
        memory_info['note'] = 'Memory monitoring not available for CPU'
    
    return memory_info


def clear_cache(device=None):
    """
    Clear device cache to free up memory.
    
    Args:
        device: Device to clear cache for (if None, uses current device)
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    elif device.type == 'mps':
        try:
            torch.mps.empty_cache()
            print("MPS cache cleared")
        except:
            print("MPS cache clearing not available")
    else:
        print("No cache to clear for CPU")


def print_device_info():
    """Print comprehensive device information."""
    info = get_device_info()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {info['mps_available']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA device count: {info['cuda_device_count']}")
        for i, name in enumerate(info['cuda_device_names']):
            print(f"  Device {i}: {name}")
    
    print(f"Selected device: {info['selected_device']}")
    print("=" * 50)


if __name__ == "__main__":
    # Demo the device utilities
    print_device_info()
    
    # Test device detection
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Test memory info
    memory = get_memory_usage(device)
    print(f"Memory info: {memory}")