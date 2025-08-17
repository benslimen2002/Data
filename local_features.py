import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import numpy as np
from PIL import Image
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LocalContentExtractor(nn.Module):
    """
    Extracts Local Content features using VGG19 backbone and GRU
    Plc = {lct-m_i, lct-m+1_i, ..., lct_i}
    """
    def __init__(self, feature_dim=512, hidden_size=128, num_layers=1):
        super().__init__()
        
        # Load pre-trained VGG19 with ImageNet weights
        self.vgg_backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Remove classification layers, keep only feature extraction
        self.vgg_backbone = nn.Sequential(*list(self.vgg_backbone.features.children())[:-1])
        
        # Add max pooling as suggested in the paper
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Final projection layer
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, pedestrian_images):
        """
        Args:
            pedestrian_images: [batch_size, seq_len, 3, 224, 224] - cropped pedestrian images
        Returns:
            local_content_features: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = pedestrian_images.shape[:2]
        
        # Reshape for batch processing
        images_flat = pedestrian_images.view(-1, 3, 224, 224)
        
        # Extract features using VGG19
        with torch.no_grad():  # Freeze VGG weights during training
            features = self.vgg_backbone(images_flat)  # [batch_size*seq_len, 512, H, W]
        
        # Apply max pooling
        pooled_features = self.max_pool(features).squeeze(-1).squeeze(-1)  # [batch_size*seq_len, 512]
        
        # Reshape back to sequence
        sequence_features = pooled_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 512]
        
        # Process temporal dimension with GRU
        gru_out, _ = self.gru(sequence_features)  # [batch_size, seq_len, hidden_size]
        
        # Apply final projection
        output = self.projection(gru_out)
        
        return output

class LocalMotionExtractor(nn.Module):
    """
    Extracts Local Motion features using Conv3D and GRU
    Plm = {lmt-m_i, lmt-m+1_i, ..., lmt_i}
    """
    def __init__(self, input_channels=2, feature_dim=512, hidden_size=128, num_layers=1):
        super().__init__()
        
        # Conv3D for spatio-temporal feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(256, feature_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU()
        )
        
        # 3D Max Pooling with kernel size 4x4 as specified
        self.mp3d = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        
        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Final projection layer
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, optical_flow_sequence):
        """
        Args:
            optical_flow_sequence: [batch_size, seq_len, 2, 224, 224] - optical flow within pedestrian bounding box
        Returns:
            local_motion_features: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = optical_flow_sequence.shape[:2]
        
        # Reshape for Conv3D: [batch_size, channels, seq_len, height, width]
        flow_reshaped = optical_flow_sequence.permute(0, 2, 1, 3, 4)
        
        # Apply Conv3D
        conv_features = self.conv3d(flow_reshaped)  # [batch_size, feature_dim, seq_len, H, W]
        
        # Apply 3D Max Pooling
        pooled_features = self.mp3d(conv_features)  # [batch_size, feature_dim, seq_len, H//4, W//4]
        
        # Global average pooling over spatial dimensions
        spatial_dims = pooled_features.shape[3:]
        global_pool = F.adaptive_avg_pool3d(pooled_features, (seq_len, 1, 1))
        global_pool = global_pool.squeeze(-1).squeeze(-1)  # [batch_size, feature_dim, seq_len]
        
        # Transpose for GRU: [batch_size, seq_len, feature_dim]
        sequence_features = global_pool.transpose(1, 2)
        
        # Process temporal dimension with GRU
        gru_out, _ = self.gru(sequence_features)  # [batch_size, seq_len, hidden_size]
        
        # Apply final projection
        output = self.projection(gru_out)
        
        return output

class PedestrianCropper:
    """
    Utility class to crop pedestrian images based on bounding box coordinates
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def crop_and_warp(self, image, bbox):
        """
        Crop pedestrian based on bounding box and warp to target size
        
        Args:
            image: PIL Image or numpy array
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            cropped_image: PIL Image of size target_size
        """
        # Handle different image types
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL - ensure values are in [0, 1] range
            if image.dim() == 3:  # [C, H, W]
                image = image.permute(1, 2, 0)  # [H, W, C]
            
            # Clamp values to [0, 1] range
            image = torch.clamp(image, 0, 1)
            
            # Convert to uint8 for PIL
            image = (image * 255).byte().cpu().numpy()
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            # Ensure numpy array is in correct format
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            # If it's something else, try to convert to PIL
            try:
                image = Image.fromarray(np.array(image))
            except:
                # Fallback: create a black image
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Crop using bounding box
        cropped = image.crop(bbox)
        
        # Resize to target size (warp)
        warped = cropped.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return warped
    
    def extract_optical_flow_region(self, optical_flow, bbox):
        """
        Extract optical flow within pedestrian bounding box
        
        Args:
            optical_flow: numpy array of optical flow
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            cropped_flow: cropped optical flow of size target_size
        """
        # Crop optical flow using bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cropped_flow = optical_flow[y1:y2, x1:x2]
        
        # Resize to target size
        cropped_flow = cv2.resize(cropped_flow, self.target_size)
        
        return cropped_flow

class LocalFeaturesProcessor(nn.Module):
    """
    Main processor that combines Local Content and Local Motion features
    """
    def __init__(self, feature_dim=512, hidden_size=128, num_layers=1):
        super().__init__()
        
        self.local_content_extractor = LocalContentExtractor(feature_dim, hidden_size, num_layers)
        self.local_motion_extractor = LocalMotionExtractor(2, feature_dim, hidden_size, num_layers)
        
        # Fusion layer to combine both features
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, pedestrian_images, optical_flow_sequence):
        """
        Args:
            pedestrian_images: [batch_size, seq_len, 3, 224, 224]
            optical_flow_sequence: [batch_size, seq_len, 2, 224, 224]
            
        Returns:
            combined_features: [batch_size, seq_len, hidden_size]
        """
        # Extract Local Content features
        local_content = self.local_content_extractor(pedestrian_images)
        
        # Extract Local Motion features
        local_motion = self.local_motion_extractor(optical_flow_sequence)
        
        # Concatenate features along feature dimension
        combined = torch.cat([local_content, local_motion], dim=-1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined)
        
        return fused_features
