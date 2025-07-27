import torch
import torch.nn as nn

from encoder import Encoder
from decoder import HierarchicalDecoder
from device_utils import get_device, setup_model_for_device

class VAE(nn.Module):
    def __init__(self, encoder_config, decoder_config, device=None):
        super(VAE, self).__init__()
    
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.device = device if device is not None else get_device()
        
        self.encoder = Encoder(input_size=encoder_config['input_size'])
        self.decoder = HierarchicalDecoder(latent_dim=decoder_config['latent_dim'], output_size=decoder_config['output_size'])
        
        # Setup model for optimal device performance
        self.encoder = setup_model_for_device(self.encoder, self.device)
        self.decoder = setup_model_for_device(self.decoder, self.device)
        
    def forward(self, input_sequence):
        # Ensure input is on the correct device
        if input_sequence.device != self.device:
            input_sequence = input_sequence.to(self.device)
            
        mu, sigma, z = self.encoder(input_sequence)
        output = self.decoder(z, input_sequence)
        return mu, sigma, z, output
    
    def to(self, device):
        """Override to method to update internal device tracking."""
        self.device = device
        return super().to(device)
