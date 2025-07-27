#!/usr/bin/env python3
"""
Example script demonstrating M1/Apple Silicon optimizations in MusicVAE.

This script shows how to:
1. Use the optimized device detection
2. Initialize models with device support
3. Train with automatic device management
4. Monitor performance and memory usage

Run this script to see the optimizations in action!
"""

import torch
import torch.optim as optim
import numpy as np
from time import time

from device_utils import (
    get_device, print_device_info, get_memory_usage, 
    clear_cache, setup_model_for_device
)
from vae import VAE
from loss import ELBO_loss


def create_sample_data(batch_size=4, seq_length=64, input_size=27, device='cpu'):
    """Create sample data for demonstration."""
    # Generate random music-like data (values between 0 and 1)
    data = torch.rand(batch_size, seq_length, input_size, device=device)
    return data


def demonstrate_device_optimization():
    """Demonstrate the device optimization features."""
    print("🎵 MusicVAE M1/Apple Silicon Optimization Demo 🎵")
    print("=" * 60)
    
    # 1. Device Detection
    print("\n1️⃣ DEVICE DETECTION")
    print("-" * 30)
    print_device_info()
    device = get_device()
    
    # 2. Model Creation with Optimization
    print("\n2️⃣ MODEL CREATION WITH OPTIMIZATION")
    print("-" * 30)
    
    encoder_config = {'input_size': 27}
    decoder_config = {'latent_dim': 512, 'output_size': 27}
    
    print("Creating optimized MusicVAE model...")
    start_time = time()
    model = VAE(encoder_config, decoder_config, device=device)
    creation_time = time() - start_time
    
    print(f"✓ Model created in {creation_time:.4f} seconds")
    print(f"✓ Model automatically placed on: {model.device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # 3. Memory Usage Monitoring
    print("\n3️⃣ MEMORY USAGE MONITORING")
    print("-" * 30)
    memory_info = get_memory_usage(device)
    print(f"Memory usage: {memory_info}")
    
    # 4. Performance Comparison
    print("\n4️⃣ PERFORMANCE DEMONSTRATION")
    print("-" * 30)
    
    batch_sizes = [1, 2, 4, 8]
    seq_length = 64
    input_size = 27
    
    print("Testing forward pass performance with different batch sizes:")
    
    model.eval()
    with torch.no_grad():
        for batch_size in batch_sizes:
            # Create sample data
            sample_data = create_sample_data(batch_size, seq_length, input_size, device)
            
            # Warmup
            for _ in range(3):
                _ = model(sample_data)
            
            # Benchmark
            start_time = time()
            num_runs = 5
            for _ in range(num_runs):
                mu, sigma, z, output = model(sample_data)
            avg_time = (time() - start_time) / num_runs
            
            samples_per_second = batch_size / avg_time
            print(f"  Batch size {batch_size:2d}: {avg_time:.4f}s/batch, {samples_per_second:.2f} samples/sec")
    
    # 5. Training Demonstration
    print("\n5️⃣ TRAINING DEMONSTRATION")
    print("-" * 30)
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    beta = 0.2
    
    print("Running mini training loop...")
    
    # Create training data
    train_data = create_sample_data(4, seq_length, input_size, device)
    
    losses = []
    for epoch in range(5):
        start_time = time()
        
        # Forward pass
        mu, sigma, z, output = model(train_data)
        
        # Compute loss
        loss, kl, adjusted_kl = ELBO_loss(output, train_data, mu, sigma, beta)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_time = time() - start_time
        losses.append(loss.item())
        
        print(f"  Epoch {epoch+1}/5: Loss={loss.item():.4f}, Time={epoch_time:.4f}s")
    
    print(f"✓ Training completed! Final loss: {losses[-1]:.4f}")
    
    # 6. Memory Cleanup
    print("\n6️⃣ MEMORY CLEANUP")
    print("-" * 30)
    
    print("Memory before cleanup:")
    memory_before = get_memory_usage(device)
    print(f"  {memory_before}")
    
    # Clear cache
    clear_cache(device)
    
    print("Memory after cleanup:")
    memory_after = get_memory_usage(device)
    print(f"  {memory_after}")
    
    # 7. Summary
    print("\n🎉 OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"✓ Device: {device}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Model creation time: {creation_time:.4f}s")
    print(f"✓ Training performance: {len(losses)} epochs completed")
    print(f"✓ Memory management: Available")
    
    if device.type == 'mps':
        print("🚀 Running on Apple Silicon with MPS acceleration!")
        print("   Performance optimized for M1/M2/M3 chips")
    elif device.type == 'cuda':
        print("🚀 Running on NVIDIA GPU with CUDA acceleration!")
    else:
        print("⚡ Running on CPU (install PyTorch with MPS/CUDA for acceleration)")
    
    print("\n" + "=" * 60)
    print("Demo completed! 🎵 Your MusicVAE is optimized and ready to use!")


def compare_device_performance():
    """Compare performance across available devices (if multiple available)."""
    print("\n🔍 DEVICE PERFORMANCE COMPARISON")
    print("=" * 60)
    
    available_devices = ['cpu']
    
    # Check for other available devices
    if torch.cuda.is_available():
        available_devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')
    
    if len(available_devices) == 1:
        print(f"Only {available_devices[0]} available for comparison")
        return
    
    print(f"Comparing performance across: {available_devices}")
    
    encoder_config = {'input_size': 27}
    decoder_config = {'latent_dim': 256, 'output_size': 27}  # Smaller model for comparison
    
    results = {}
    
    for device_name in available_devices:
        device = torch.device(device_name)
        print(f"\nTesting {device_name.upper()}...")
        
        # Create model
        model = VAE(encoder_config, decoder_config, device=device)
        model.eval()
        
        # Create test data
        test_data = torch.rand(4, 64, 27, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_data)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time()
                _ = model(test_data)
                times.append(time() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[device_name] = {'avg_time': avg_time, 'std_time': std_time}
        
        print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    
    # Show comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("-" * 30)
    baseline = results['cpu']['avg_time']
    
    for device_name, stats in results.items():
        speedup = baseline / stats['avg_time']
        print(f"{device_name.upper():>6}: {stats['avg_time']:.4f}s (x{speedup:.2f} vs CPU)")


if __name__ == "__main__":
    try:
        demonstrate_device_optimization()
        compare_device_performance()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()