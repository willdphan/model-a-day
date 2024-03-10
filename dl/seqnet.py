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

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1)

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
    

# class FeatureProcessor(nn.Module):
#     def __init__(self, inDims, outDims):
#         super(FeatureProcessor, self).__init__()
#         self.fc1 = nn.Linear(inDims, 1024)  # Example dimension reduction
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(1024, outDims)  # Output layer, adjust `outDims` as needed

#     def forward(self, x):
#         # Ensure input x is properly flattened in case it's not
#         x = x.view(-1, 4096)  # Adjust for NetVLAD output dimension
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# if __name__ == '__main__':
#     inDims = 4096  # From NetVLAD features
#     outDims = 256  # Example output dimension, adjust based on your requirements
#     model = FeatureProcessor(inDims, outDims)

#     # Example input tensor
#     input_tensor = torch.randn(1, 4096)  # Simulating NetVLAD output

#     # Process the feature vector
#     output = model(input_tensor)
#     print("Output Shape:", output.shape)


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