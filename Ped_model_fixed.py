import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, PegasusForConditionalGeneration
from peft import get_peft_model, LoraConfig, LoftQConfig
from peft.utils import print_trainable_parameters
from modules.local_features import LocalFeaturesProcessor
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PedVLMT5(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Make tokenizer and text model
        if config.lm == 'T5-Base':
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
            hidden_size = self.model.config.d_model
            vac_size = self.model.config.vocab_size
        elif config.lm == 'GPT':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            hidden_size = 768  # GPT_N_EMBED
            vac_size = self.model.config.vocab_size
        elif config.lm == 'PN':
            self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
            hidden_size = self.model.config.d_model
            vac_size = self.model.config.vocab_size
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')
            hidden_size = self.model.config.d_model
            vac_size = self.model.config.vocab_size

        if config.lora:
            if config.lm == 'T5-Base':
                tm = ['q', 'v']
            elif config.lm == 'BERT':
                tm = ['query', 'key', 'value']
            else:
                tm = ['c_attn']

            # For quantization
            loftq_config = LoftQConfig(loftq_bits=8)

            # Create LoRA model
            lora_config = LoraConfig(
                r=config.lora_dim,
                lora_alpha=config.lora_alpha,
                loftq_config=loftq_config,
                lora_dropout=config.lora_dropout,
                bias='none',
                target_modules=tm
            )
            self.model = get_peft_model(self.model, lora_config)

        if config.freeze_lm:
            for p in self.model.parameters():
                p.requires_grad = False

        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        # Create instance for multi-view processor
        self.mvp = self.ImageProcessor(config.gpa_hidden_size, hidden_size, config.lm, config.attention, 768, config.num_head, config.encoder, freeze=True)
        self.bodypose_processor = self.BodyPoseProcessor(hidden_size)
        self.int = self.PedestrianCrossingClassifier(vac_size)
        
        # Local features processor
        self.use_local_features = getattr(config, 'use_local_features', True)
        if self.use_local_features:
            self.local_features_processor = LocalFeaturesProcessor(
                feature_dim=512, 
                hidden_size=128, 
                num_layers=1
            )

    class BodyPoseProcessor(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            
            # Process body pose keypoints (17 keypoints with x,y coordinates = 34 features)
            self.pose_encoder = nn.Sequential(
                nn.Linear(34, 128),  # 17 keypoints * 2 coordinates = 34
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, hidden_size)
            )
            
            # Attention mechanism for keypoints
            self.keypoint_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            
            # Final projection to match hidden size
            self.final_projection = nn.Linear(hidden_size, hidden_size)
            
        def forward(self, bodypose):
            # bodypose shape: [batch_size, 17, 2]
            batch_size = bodypose.shape[0]
            
            # Flatten keypoint coordinates: [batch_size, 34]
            flattened_pose = bodypose.view(batch_size, -1)
            
            # Encode pose features: [batch_size, hidden_size]
            encoded_pose = self.pose_encoder(flattened_pose)
            
            # Reshape for attention: [batch_size, 1, hidden_size]
            encoded_pose = encoded_pose.unsqueeze(1)
            
            # Apply self-attention
            attended_pose, _ = self.keypoint_attention(encoded_pose, encoded_pose, encoded_pose)
            
            # Final projection
            output = self.final_projection(attended_pose.squeeze(1))  # [batch_size, hidden_size]
            
            return output

    class PedestrianCrossingClassifier(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(vocab_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)  # 2 classes: crossing or not crossing
            )
            
        def forward(self, logits):
            # logits shape: [batch_size, seq_len, vocab_size]
            # We'll use the last token's logits for classification
            last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            output = self.classifier(last_token_logits)
            return output

    class ImageProcessor(nn.Module):
        def __init__(self, gpa_hidden_size, hidden_size, lm, attention, VIT_HIDDEN_STATE, num_head, encoder, freeze=True):
            super().__init__()
            self.attention = attention
            self.lm = lm
            self.encoder = encoder
            
            if encoder == 'clip':
                from transformers import CLIPProcessor, CLIPModel
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                if freeze:
                    for param in self.clip_model.parameters():
                        param.requires_grad = False
                self.image_projection = nn.Linear(512, hidden_size)
            elif encoder == 'vit':
                from transformers import ViTImageProcessor, ViTModel
                self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
                if freeze:
                    for param in self.vit_model.parameters():
                        param.requires_grad = False
                self.image_projection = nn.Linear(768, hidden_size)
            else:  # resnet50
                import torchvision.models as models
                self.resnet = models.resnet50(pretrained=True)
                if freeze:
                    for param in self.resnet.parameters():
                        param.requires_grad = False
                self.image_projection = nn.Linear(2048, hidden_size)
            
            # Multi-view attention
            if attention:
                self.atten = self.MultiViewAttention(gpa_hidden_size, hidden_size, num_head)
            
        def forward(self, text_enc, imgs, model):
            batch_size = imgs.shape[0]
            num_frames = imgs.shape[1]
            
            # Process each frame
            frame_embeddings = []
            for i in range(num_frames):
                frame = imgs[:, i, :, :, :]  # [batch_size, channels, height, width]
                
                if self.encoder == 'clip':
                    # Use CLIP for image processing
                    inputs = self.clip_processor(images=frame, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        image_features = self.clip_model.get_image_features(**inputs)
                    frame_embedding = self.image_projection(image_features)
                elif self.encoder == 'vit':
                    # Use ViT for image processing
                    inputs = self.vit_processor(images=frame, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.vit_model(**inputs)
                    frame_embedding = self.image_projection(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
                else:  # resnet50
                    # Use ResNet for image processing
                    with torch.no_grad():
                        features = self.resnet(frame)
                    frame_embedding = self.image_projection(features)
                
                frame_embeddings.append(frame_embedding)
            
            # Stack frame embeddings: [batch_size, num_frames, hidden_size]
            imgs_embedding = torch.stack(frame_embeddings, dim=1)
            
            # Process text embeddings
            if self.lm == 'T5-Base':
                text_embeddings = model.encoder.embed_tokens(text_enc)
            elif self.lm == 'GPT':
                text_embeddings = model.transformer.wte(text_enc)
            else:
                text_embeddings = model.model.encoder.embed_tokens(text_enc)
            
            # Apply multi-view attention if enabled
            if self.attention:
                imgs_embedding, text_embeddings = self.atten(imgs_embedding, text_embeddings)
            
            # Concatenate embeddings -> (batch_size x (num_frames + text_len) x hidden_size)
            merged_embedding = torch.cat([imgs_embedding, text_embeddings], dim=1)
            
            return merged_embedding

    class MultiViewAttention(nn.Module):
        def __init__(self, gpa_hidden_size, hidden_size, num_head):
            super().__init__()
            self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=num_head, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, gpa_hidden_size),
                nn.ReLU(),
                nn.Linear(gpa_hidden_size, hidden_size)
            )
            
        def forward(self, imgs_embedding, text_embeddings):
            # Cross attention between image and text
            attended_imgs, _ = self.cross_attention(imgs_embedding, text_embeddings, text_embeddings)
            attended_imgs = self.norm1(attended_imgs + imgs_embedding)
            
            attended_text, _ = self.cross_attention(text_embeddings, imgs_embedding, imgs_embedding)
            attended_text = self.norm2(attended_text + text_embeddings)
            
            return attended_imgs, attended_text

    def forward(self, text_enc, imgs, labels=None, bodypose=None, local_content_features=None, local_motion_features=None):
        # Get the merged embeddings
        merged_embedding = self.mvp(text_enc, imgs, self.model)
        
        # Process body pose if available
        if bodypose is not None and bodypose.numel() > 0:
            pose_features = self.bodypose_processor(bodypose)
            # Add pose features to the merged embedding
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                pose_embedding = pose_features.unsqueeze(1).expand(-1, 49, -1)
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + pose_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                pose_embedding = pose_features.unsqueeze(1).expand(-1, merged_embedding.shape[1], -1)
                merged_embedding = merged_embedding + pose_embedding

        # Process local features if available
        if self.use_local_features and local_content_features is not None and local_motion_features is not None:
            # Combine local content and motion features
            local_features = torch.cat([local_content_features, local_motion_features], dim=-1)  # [batch_size, 256]
            
            # Project to match hidden size
            if hasattr(self, 'local_features_projection'):
                local_features = self.local_features_projection(local_features)
            else:
                self.local_features_projection = nn.Linear(256, merged_embedding.shape[-1]).to(device)
                local_features = self.local_features_projection(local_features)
            
            # Add local features to the merged embedding
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                # Ensure local_features has correct shape [batch_size, hidden_size]
                if len(local_features.shape) == 4:  # If [batch_size, 1, 1, hidden_size]
                    local_features = local_features.squeeze(1).squeeze(1)  # Remove extra dimensions
                local_embedding = local_features.unsqueeze(1).expand(-1, 49, -1)
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + local_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                # Ensure local_features has correct shape [batch_size, hidden_size]
                if len(local_features.shape) == 4:  # If [batch_size, 1, 1, hidden_size]
                    local_features = local_features.squeeze(1).squeeze(1)  # Remove extra dimensions
                local_embedding = local_features.unsqueeze(1).expand(-1, merged_embedding.shape[1], -1)
                merged_embedding = merged_embedding + local_embedding

        # If training include the labels
        out_t5 = self.model(inputs_embeds=merged_embedding, labels=labels)
        int_output = self.int(out_t5.logits)
        return out_t5, int_output

    def generate(self, text_enc, imgs, bodypose=None, lidar=None, local_content_features=None, local_motion_features=None):
        merged_embedding = self.mvp(text_enc, imgs, self.model)
        
        # Process body pose if available
        if bodypose is not None and bodypose.numel() > 0:
            pose_features = self.bodypose_processor(bodypose)
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                pose_embedding = pose_features.unsqueeze(1).expand(-1, 49, -1)
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + pose_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                pose_embedding = pose_features.unsqueeze(1).expand(-1, merged_embedding.shape[1], -1)
                merged_embedding = merged_embedding + pose_embedding

        # Process local features if available
        if self.use_local_features and local_content_features is not None and local_motion_features is not None:
            # Combine local content and motion features
            local_features = torch.cat([local_content_features, local_motion_features], dim=-1)  # [batch_size, 256]
            
            # Project to match hidden size
            if hasattr(self, 'local_features_projection'):
                local_features = self.local_features_projection(local_features)
            else:
                self.local_features_projection = nn.Linear(256, merged_embedding.shape[-1]).to(device)
                local_features = self.local_features_projection(local_features)
            
            # Add local features to the merged embedding
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                # Ensure local_features has correct shape [batch_size, hidden_size]
                if len(local_features.shape) == 4:  # If [batch_size, 1, 1, hidden_size]
                    local_features = local_features.squeeze(1).squeeze(1)  # Remove extra dimensions
                local_embedding = local_features.unsqueeze(1).expand(-1, 49, -1)
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + local_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                # Ensure local_features has correct shape [batch_size, hidden_size]
                if len(local_features.shape) == 4:  # If [batch_size, 1, 1, hidden_size]
                    local_features = local_features.squeeze(1).squeeze(1)  # Remove extra dimensions
                local_embedding = local_features.unsqueeze(1).expand(-1, merged_embedding.shape[1], -1)
                merged_embedding = merged_embedding + local_embedding

        attention_mask = torch.ones(merged_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((merged_embedding.shape[0], 1), dtype=torch.long, device=device) * self.model.config.decoder_start_token_id
        output_ids = self.model.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=merged_embedding, max_length=512, early_stopping=True)
        
        return output_ids
