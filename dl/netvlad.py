"""
NetVLAD
Created: 03-07-2024
---
Aggregates CNN features into a compact representation by clustering them into predefined clusters and computing the residuals relative to cluster centers. Enhances the discriminability of features for place recognition tasks.

EmbedNet: 

Combines a base CNN model with the NetVLAD layer to extract and aggregate deep features from images into a single compact descriptor.

TripletNet: 

Utilizes EmbedNet to process triplets of images (anchor, positive, negative) for metric learning, aiming to minimize the distance between an anchor and a positive sample while maximizing the distance between the anchor and a negative sample.

The input dimensions [B, C, H, W] are transformed into a VLAD vector of size [B, num_clusters * C], compactly representing the input image's essential features in relation to the learned cluster centroids.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        """
        SOFT ASSIGNMENT

        In short, NetVLAD forward method first normalizes the input features, then assigns them to clusters (soft assignment) and computes the residuals between the features and their corresponding cluster centers.  

        "Soft assignment" refers to the process of distributing a data point's membership across multiple clusters, rather than assigning it to a single cluster (hard assignment). Each cluster gets a membership score that reflects the likelihood or degree to which the data point belongs to that cluster. This approach allows for more nuanced grouping, capturing the ambiguity in data relationships and often leading to better performance in tasks like clustering and feature aggregation.
        
        These residuals are aggregated and normalized to produce a compact and discriminative VLAD descriptor for each input.
        """
        N, C = x.shape[:2]
        print(f"Input shape: {x.shape}")  # Print the initial shape of the input

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
            print(f"Shape after normalization: {x.shape}")  # Shape remains unchanged but print to track

        # soft-assignment of features to clusters using a convolution layer followed by a softmax operation
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        print(f"Shape after soft assignment (before softmax): {soft_assign.shape}")
        soft_assign = F.softmax(soft_assign, dim=1)
        # Despite the shape remaining the same (torch.Size([1, 8, 49])), the values within are transformed such that for each feature (across the last dimension), the scores across the 8 clusters sum up to 1
        print(f"Shape after softmax: {soft_assign.shape}") 
        # then flattens
        x_flatten = x.view(N, C, -1)
        print(f"Shape after flattening: {x_flatten.shape}")
        
        """
        CALCULATE RESIDUALS

        Calculating residuals to each cluster involves computing the difference between the features (after flattening and expansion) and the corresponding cluster centroids. This process measures how far each feature is from every cluster center. 
        
        The residuals are then weighted by the soft assignment probabilities, indicating the degree of belongingness of each feature to the clusters.

        Calculating and aggregating residuals in NetVLAD creates a robust global descriptor for each image, significantly enhancing its performance in visual place recognition, image retrieval, and distinguishing between similar images by capturing unique spatial relationships of features.
        """
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        print(f"Shape of residual before sum: {residual.shape}")
        vlad = residual.sum(dim=-1)
        print(f"Shape of vlad before intra-normalization: {vlad.shape}")

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        print(f"Shape of vlad after intra-normalization: {vlad.shape}")
        vlad = vlad.view(x.size(0), -1)  # flatten
        print(f"Shape of vlad after flatten: {vlad.shape}")
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        print(f"Final output shape: {vlad.shape}")

        return vlad



class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        print(f"EmbedNet input shape: {x.shape}")
        x = self.base_model(x)
        print(f"After base model (CNN Processing) shape: {x.shape}")
        embedded_x = self.net_vlad(x)
        print(f"After NetVLAD shape: {embedded_x.shape}")
        return embedded_x


class TripletNet(nn.Module):
    """
    The TripletNet class uses an embedding network to process triplets of images (anchor, positive, negative) to learn discriminative features by bringing the anchor and positive images closer together in the feature space, while pushing the anchor and negative images further apart. 
    
    The point of using a TripletNet is to train a model in a way that it learns to differentiate between similar and dissimilar images effectively. By processing triplets of images (an anchor, a positive example similar to the anchor, and a negative example dissimilar from the anchor), it helps in embedding images into a space where similar images are closer together and dissimilar images are farther apart, enhancing the model's accuracy in tasks like image matching, retrieval, and classification.
    """
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        print(f"Triplet input shapes: a={a.shape}, p={p.shape}, n={n.shape}")
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        print(f"Feature extract input shape: {x.shape}")
        features = self.embed_net(x)
        print(f"Extracted features shape: {features.shape}")
        return features
    
if __name__ == '__main__':
        # Load a pretrained model and modify it
        base_model = models.resnet18(pretrained=True)
        base_model = nn.Sequential(*list(base_model.children())[:-2])
        output_feature_dim = base_model(torch.randn(1, 3, 224, 224)).shape[1]  # Get the feature dimension

        # Instantiate NetVLAD and other models
        net_vlad = NetVLAD(num_clusters=8, dim=output_feature_dim, alpha=100.0)
        embed_net = EmbedNet(base_model=base_model, net_vlad=net_vlad)
        triplet_net = TripletNet(embed_net=embed_net)

        # Move the model to the appropriate device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        triplet_net = triplet_net.to(device)

        # Simulate loading and preprocessing images
        # Creating dummy data: 1 batch with 3 images of 3x224x224
        # Would replace these with actual preprocessed images
        # [Batch Size, Number of Channels, Height of Image, Width of Image]        
        a = torch.randn(1, 3, 224, 224).to(device)  # Positive
        p = torch.randn(1, 3, 224, 224).to(device)  # Positive
        n = torch.randn(1, 3, 224, 224).to(device)  # Negative

        # Extract features
        with torch.no_grad():
            triplet_net.eval()  # Set the network to evaluation mode
            features_a, features_p, features_n = triplet_net(a, p, n)

        print(features_a.shape, features_p.shape, features_n.shape)
        # batch size, dimensionality
        # torch.Size([1, 4096]) torch.Size([1, 4096]) torch.Size([1, 4096])

