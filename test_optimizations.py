#!/usr/bin/env python3
"""
Test script for M1/Apple Silicon optimizations in MusicVAE.
This script validates that the device detection and model optimization work correctly.
"""

import torch
import torch.optim as optim
import numpy as np
from time import time

from device_utils import get_device, print_device_info, get_memory_usage, clear_cache
from vae import VAE
from loss import ELBO_loss


def test_device_detection():
    """Test device detection functionality."""
    print("=" * 60)
    print("TESTING DEVICE DETECTION")
    print("=" * 60)
    
    print_device_info()
    device = get_device()
    
    print(f"\nSelected device: {device}")
    print(f"Device type: {device.type}")
    
    # Test memory monitoring
    memory_info = get_memory_usage(device)
    print(f"Memory info: {memory_info}")
    
    return device


def test_model_creation(device):
    """Test model creation with device optimization."""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    encoder_config = {'input_size': 27}
    decoder_config = {'latent_dim': 512, 'output_size': 27}
    
    start_time = time()
    model = VAE(encoder_config, decoder_config, device=device)
    creation_time = time() - start_time
    
    print(f"✓ Model created in {creation_time:.4f} seconds")
    print(f"Model device: {model.device}")
    print(f"Encoder device: {next(model.encoder.parameters()).device}")
    print(f"Decoder device: {next(model.decoder.parameters()).device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def test_forward_pass(model, device):
    """Test forward pass with different batch sizes."""
    print("\n" + "=" * 60)
    print("TESTING FORWARD PASS")
    print("=" * 60)
    
    batch_sizes = [1, 4, 8]
    seq_length = 64
    input_size = 27
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create test input
        test_input = torch.rand(batch_size, seq_length, input_size)
        
        start_time = time()
        with torch.no_grad():
            mu, sigma, z, output = model(test_input)
        forward_time = time() - start_time
        
        print(f"  ✓ Forward pass completed in {forward_time:.4f} seconds")
        print(f"  Input shape: {test_input.shape} -> Output shape: {output.shape}")
        print(f"  All tensors on device: {device}")
        
        # Verify all outputs are on correct device
        assert mu.device == device, f"mu on wrong device: {mu.device}"
        assert sigma.device == device, f"sigma on wrong device: {sigma.device}" 
        assert z.device == device, f"z on wrong device: {z.device}"
        assert output.device == device, f"output on wrong device: {output.device}"


def test_training_step(model, device):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING STEP")
    print("=" * 60)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    beta = 0.2
    
    # Create batch data
    batch_size = 4
    seq_length = 64
    input_size = 27
    
    test_input = torch.rand(batch_size, seq_length, input_size)
    
    model.train()
    
    start_time = time()
    
    # Forward pass
    mu, sigma, z, output = model(test_input)
    
    # Compute loss
    loss, kl, adjusted_kl = ELBO_loss(output, test_input, mu, sigma, beta)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    training_time = time() - start_time
    
    print(f"✓ Training step completed in {training_time:.4f} seconds")
    print(f"Loss: {loss.item():.6f}")
    print(f"KL divergence: {kl.item():.6f}")
    print(f"Adjusted KL: {adjusted_kl.item():.6f}")
    
    return loss.item()


def test_memory_management(device):
    """Test memory management functions."""
    print("\n" + "=" * 60)
    print("TESTING MEMORY MANAGEMENT")
    print("=" * 60)
    
    # Get memory usage before
    memory_before = get_memory_usage(device)
    print(f"Memory before: {memory_before}")
    
    # Create some tensors
    large_tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000, device=device)
        large_tensors.append(tensor)
    
    memory_after = get_memory_usage(device)
    print(f"Memory after creating tensors: {memory_after}")
    
    # Clear tensors
    del large_tensors
    
    # Clear cache
    clear_cache(device)
    
    memory_cleared = get_memory_usage(device)
    print(f"Memory after clearing cache: {memory_cleared}")


def run_performance_benchmark(model, device):
    """Run a simple performance benchmark."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    batch_size = 8
    seq_length = 64
    input_size = 27
    num_iterations = 10
    
    model.eval()
    
    # Warmup
    for _ in range(3):
        test_input = torch.rand(batch_size, seq_length, input_size)
        with torch.no_grad():
            _ = model(test_input)
    
    # Benchmark forward pass
    forward_times = []
    for i in range(num_iterations):
        test_input = torch.rand(batch_size, seq_length, input_size)
        
        start_time = time()
        with torch.no_grad():
            _ = model(test_input)
        forward_time = time() - start_time
        forward_times.append(forward_time)
    
    avg_forward_time = np.mean(forward_times)
    std_forward_time = np.std(forward_times)
    
    print(f"Forward pass statistics ({num_iterations} iterations):")
    print(f"  Average time: {avg_forward_time:.4f} ± {std_forward_time:.4f} seconds")
    print(f"  Samples per second: {batch_size / avg_forward_time:.2f}")
    print(f"  Device: {device}")


def main():
    """Main test function."""
    print("MusicVAE M1/Apple Silicon Optimization Test Suite")
    print("=" * 60)
    
    try:
        # Test device detection
        device = test_device_detection()
        
        # Test model creation
        model = test_model_creation(device)
        
        # Test forward pass
        test_forward_pass(model, device)
        
        # Test training step
        test_training_step(model, device)
        
        # Test memory management
        test_memory_management(device)
        
        # Run performance benchmark
        run_performance_benchmark(model, device)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("M1/Apple Silicon optimizations are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)