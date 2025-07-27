from vae import VAE

from data import *
from dataloader import *

from torch.utils.data import Dataset, DataLoader
from loss import *
from torch import optim
import numpy as np

# Import device utilities for M1/Apple Silicon optimization
from device_utils import get_device, setup_model_for_device, print_device_info, move_to_device

# 1 bar = 16 notes, 4 bar : T = 64

#dataset config
dataset_config={
    'train':{
        'tfrecord':'./groove-v1.0.0-midionly/train_preprocess.tfrecord',
        'batch_size':256

    },
    'val':{
        'tfrecord':'./groove-v1.0.0-midionly/val_preprocess.tfrecord',
        'batch_size':100

    }, 
    'test':{
        'tfrecord':'./groove-v1.0.0-midionly/test_preprocess.tfrecord',
        'batch_size':100

    },
}
#config for the encoder of VAE
encoder_config = {
    'input_size': 27 # possible notes to play
}


#config for the decoder of VAE
decoder_config = {
    'latent_dim': 512, # As mention in the paper
    'output_size': 27 
}


#create all dataset loader
def generate_dataloader(dataset_config,shuffle = True):





    train_dataset = dataset_tensor(dataset_config['train']['tfrecord'],dataset_config['train']['batch_size'])
    val_dataset = dataset_tensor(dataset_config['val']['tfrecord'],dataset_config['val']['batch_size'])
    test_dataset = dataset_tensor(dataset_config['test']['tfrecord'],dataset_config['test']['batch_size'])

    train_dataset = CustomDataset(train_dataset)
    test_dataset = CustomDataset(test_dataset)
    val_dataset = CustomDataset(val_dataset)
    
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=dataset_config['train']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=dataset_config['test']['batch_size'])
    val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=dataset_config['val']['batch_size'])

    return train_loader, test_loader,val_loader


# Training function with device optimization
def train(model, optimizer, train_loader, num_epochs, beta, device):
    train_loss = []
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        batch_loss = []

        for batch in train_loader:
            # Move batch data to device
            input_sequence = move_to_device(batch['input_sequence'], device)
            output_sequence = move_to_device(batch['output_sequence'], device)
            control_sequence = move_to_device(batch['control_sequence'], device)
            sequence_length = move_to_device(batch['sequence_length'], device)
            
            mu, sigma, z, output = model(input_sequence)

            loss, kl, adjusted_kl = ELBO_loss(output, input_sequence, mu, sigma, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        

        #Training not yet completed. Need to add accuracy and validation

        train_loss.append(np.mean(batch_loss))
        print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {np.mean(batch_loss):.6f}')
    
    return train_loss


if __name__ == '__main__':
    # Initialize device with M1/Apple Silicon optimization
    print("Initializing MusicVAE with M1/Apple Silicon optimizations...")
    print_device_info()
    device = get_device()
    
    # Generate data loaders
    train_loader, test_loader, val_loader = generate_dataloader(dataset_config)

    # Initialize model with device support
    model = VAE(encoder_config, decoder_config, device=device)
    print(f"Model initialized and moved to device: {device}")

    # Training parameters
    num_epochs = 200
    beta = 0.2
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    total_loss = train(model, optimizer, train_loader, num_epochs, beta, device)
