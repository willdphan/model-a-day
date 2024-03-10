"""
SeqNet
Created: 03-07-2024
---
seqNet: Applies 1D convolution to input data, reducing its sequence dimension through averaging. It outputs a 2D tensor [Batch, OutDims] representing aggregated features.

Delta: Calculates weighted sums across the sequence dimension, outputting a 2D tensor [Batch, Channels] that captures changes across the sequence.
"""

# https://github.com/oravus/seqNet/blob/main/seqNet.py
import torch
import torch.nn as nn
import numpy as np

class FeatureMixer(nn.Module):
    def __init__(self, channels, num_blocks):
        super(FeatureMixer, self).__init__()
        self.num_blocks = num_blocks
        self.mlp_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(channels, channels),  # W_1
            nn.ReLU(),                      # ReLU activation
            nn.Linear(channels, channels)   # W_2
        ) for _ in range(num_blocks)])

    def forward(self, x):
        # Original shape of x: [B, C, T] where T might have changed due to convolution
        B, C, T = x.shape
        
        # We reshape x to treat each time step's features across all channels as independent inputs to the MLP
        x = x.view(B * T, C)  # Reshape to [B*T, C]
        
        for mlp in self.mlp_blocks:
            x_res = mlp(x)
            x = x + x_res  # Skip connection
            
        # After processing, reshape x back to its original shape [B, C, T]
        x = x.view(B, C, T)
        
        return x

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5, num_mixer_blocks=1):
        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)
        self.feature_mixer = FeatureMixer(outDims, num_mixer_blocks)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)  # convert [B,C] to [B,1,C]
        x = x.permute(0, 2, 1)  # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = self.feature_mixer(seqFt)  # Pass through FeatureMixer
        seqFt = torch.mean(seqFt, -1)
        return seqFt

    
class Delta(nn.Module):
    def __init__(self, inDims, seqL):

        super(Delta, self).__init__()
        self.inDims = inDims
        self.weight = (np.ones(seqL,np.float32))/(seqL/2.0)
        self.weight[:seqL//2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(self.weight),requires_grad=False)

    def forward(self, x):

        # make desc dim as C
        x = x.permute(0,2,1) # makes [B,T,C] as [B,C,T]
        delta = torch.matmul(x,self.weight)

        return delta
    
if __name__ == '__main__':
    # Parameters for the model (replace these with actual values you need)
    inDims = 10  # Input feature dimensions
    outDims = 5  # Output feature dimensions
    seqL = 20    # Sequence length
    batch_size = 3

    # Create instances of the models
    seq_net_model = seqNet(inDims=inDims, outDims=outDims, seqL=seqL)
    delta_model = Delta(inDims=inDims, seqL=seqL)

    # Create dummy input data
    input_data = torch.randn(batch_size, seqL, inDims)

    # Forward pass through seqNet
    seq_net_output = seq_net_model(input_data)

    # Forward pass through Delta
    delta_output = delta_model(input_data)

    print("SeqNet Output:", seq_net_output)
    print("SeqNet Output Shape:", seq_net_output.shape)

    print("Delta Output:", delta_output)
    print("Delta Output Shape:", delta_output.shape)