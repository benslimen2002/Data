from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer, GPT2LMHeadModel, GPT2TokenizerFast,BertForSequenceClassification,PegasusForConditionalGeneration, CLIPProcessor, CLIPModel
from torchvision.models import vit_b_32
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model, LoftQConfig
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50,ResNet50_Weights
from .local_features import LocalFeaturesProcessor

VIT_HIDDEN_STATE = 768
VIT_SEQ_LENGTH = 49
GPT_N_EMBED = 1024
TEXT_MAX_LENGTH = 463
NUM_HEAD=8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class PedVLMT5(nn.Module):

    def __init__(self, config):

        super().__init__()

        # Make tokenizer and text model
        if config.lm == 'T5-Base':
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
            hidden_size = self.model.config.d_model
            vac_size=self.model.config.vocab_size
        elif config.lm == 'GPT':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            # for name, module in self.model.named_modules():
            #     print(name)
            hidden_size = GPT_N_EMBED
        elif config.lm == 'PN':
            
            self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
            hidden_size = self.model.config.d_model
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')
            hidden_size = self.model.config.d_model
            vac_size=self.model.config.vocab_size
        

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

        # hidden_size = self.model.config.d_model

        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        # Create instance for multi-view processor
        

        self.mvp = self.ImageProcessor(config.gpa_hidden_size, hidden_size, config.lm,config.attention,VIT_HIDDEN_STATE,config.num_head,config.encoder,freeze=True)
        self.bodypose_processor = self.BodyPoseProcessor(hidden_size)
        self.int=self.PedestrianCrossingClassifier(vac_size)
        
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
            pose_features = self.pose_encoder(flattened_pose)
            
            # Reshape for attention: [batch_size, 1, hidden_size]
            pose_features = pose_features.unsqueeze(1)
            
            # Apply self-attention to keypoints
            attended_pose, _ = self.keypoint_attention(pose_features, pose_features, pose_features)
            
            # Final projection
            output = self.final_projection(attended_pose.squeeze(1))  # [batch_size, hidden_size]
            
            return output
       
    # class PedestrianClassifier(nn.Module):
    #     def __init__(self, hidden_size):
    #         super().__init__()
    #         self.pooling = nn.AdaptiveAvgPool1d(1)
    #         self.fc1 = nn.Linear(hidden_size, 256)
    #         self.fc2 = nn.Linear(256, 64)
    #         self.fc3 = nn.Linear(64, 2)  # 2 classes: crossing or not crossing
    #         self.relu = nn.ReLU()

    #     def forward(self, logits):
    #         pooled = self.pooling(logits.transpose(1, 2)).squeeze(2)
    #         x = self.relu(self.fc1(pooled))
    #         x = self.relu(self.fc2(x))
    #         output = self.fc3(x)
    #         return output
    class PedestrianCrossingClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=256):
            super().__init__()
            self.pooling = nn.AdaptiveAvgPool1d(1)  # Pooling over sequence length
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 64)
            self.fc3 = nn.Linear(64, 2)  # 2 classes: crossing or not crossing
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1) 

        def forward(self, logits):
            # logits shape: [batch_size, sequence_length, vocab_size]
            x = logits.transpose(1, 2)  # Shape: [batch_size, vocab_size, sequence_length]
            x = self.pooling(x).squeeze(2)  # Shape: [batch_size, vocab_size]
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            output = self.softmax(x) 
            return output
 
    class ImageProcessor(nn.Module):

        def __init__(self, gpa_hidden_size, hidden_size, lm,attention,embed_dim, num_heads,encoder,freeze=False):

            super().__init__()

            # Use ViT for image embeddings
            self.lm = lm
            self.attention=attention
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.encoder = encoder
            if self.encoder== 'clip':
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.img_projection_layer = nn.Linear(in_features=512, out_features=hidden_size)
            elif self.encoder=='resent50':
                self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
                # model.eval()  # Set to evaluation mode

                # Remove the last fully connected layer
                self.img_model = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.img_projection_layer = nn.Linear(in_features=2048, out_features=hidden_size)

            else:
                self.img_model = vit_b_32(weights='DEFAULT')
            # Modal embedding to distinguish between image and text
            self.modal_embeddings = nn.Embedding(2, hidden_size)
            self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)

            # If we are freezing the CLIP embeddings
            if freeze:
                for param in self.img_model.parameters():
                    param.requires_grad = False

            # Set matrices based on MIVC paper
            self.w = nn.Linear(in_features=gpa_hidden_size, out_features=1)
            self.Z = nn.Sequential(
                nn.Linear(in_features=VIT_HIDDEN_STATE * VIT_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
                nn.Tanh()
            )
            self.G = nn.Sequential(
                nn.Linear(in_features=VIT_HIDDEN_STATE * VIT_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
                nn.Sigmoid()
            )
            if self.lm not in ['T5-Base', 'T5-Large', ]:
                self.img_projection_layer = nn.Linear(in_features=VIT_HIDDEN_STATE, out_features=hidden_size)
            
            # for attention 
            if self.attention== True:
                self.image_proj = nn.Linear(embed_dim, embed_dim)
                self.text_proj = nn.Linear(embed_dim, embed_dim)
                self.output_proj = nn.Linear(embed_dim, embed_dim)
                
                self.mha = nn.MultiheadAttention(embed_dim, num_heads)
           

            if self.lm != 'T5-Base':
                self.img_projection_layer = nn.Linear(in_features=VIT_HIDDEN_STATE, out_features=hidden_size)

        def gpa(self, img_embeddings):

            """"
            Calculates the gated-pooling attention score for the image embeddings
            :param img_embeddings: (6x768) dimensional
            :return single embedding of size (768,)
            """

            # Get weights for gated pooling attention
            gpa_weights = torch.softmax(self.w(self.Z(img_embeddings) * self.G(img_embeddings)), dim=0)

            # Take a linear combination of all the image embeddings
            fused_embeddings = torch.sum(gpa_weights * img_embeddings, dim=0)

            return fused_embeddings
        
        def atten(self, image_features, text_features):
        # image_features: [batch, 49, 768]
        # text_features: [batch, x, 768]
        
            batch_size, img_seq_len, _ = image_features.shape
            _, txt_seq_len, _ = text_features.shape
            
            # Project image and text features
            image_proj = self.image_proj(image_features)
            text_proj = self.text_proj(text_features)
            
            # Concatenate image and text features
            combined_features = torch.cat([image_proj, text_proj], dim=1)
            
            # Prepare for multi-head attention
            combined_features = combined_features.transpose(0, 1)  # [seq_len, batch, embed_dim]
            
            # Self-attention
            attn_output, _ = self.mha(combined_features, combined_features, combined_features)
            
            # Project output
            output = self.output_proj(attn_output.transpose(0, 1))
            
            # Split back to image and text
            image_output = output[:, :img_seq_len, :]
            text_output = output[:, img_seq_len:, :]
            
            return image_output, text_output

        def get_img_embedding(self, imgs):

            N = imgs.shape[0]  #batch size=4

            # Process into patches (N x 6 x 49 x H) input=[4,6,3,224,224]
            if self.encoder == 'clip':
                # img_process=torch.stack([self.clip_processor(images=img, return_tensors="pt") for img in imgs], dim=0)
                with torch.no_grad():
                    
                    merged_embedding = torch.stack([self.img_model.get_image_features(img) for img in imgs], dim=0) #output [4,2,512]
                merged_embedding = self.img_projection_layer(merged_embedding)
            elif self.encoder=='resent50':
                merged_embedding = torch.stack([self.img_model(img) for img in imgs], dim=0) #output [4,2,2048]
                merged_embedding = self.img_projection_layer(merged_embedding.squeeze())
            else:
                merged_embedding = torch.stack([self.img_model._process_input(img) for img in imgs], dim=0) #output [4,6,49,768]
                # merged_embedding = self.img_projection_layer(merged_embedding)
            

            # Concatenate the batch class tokens -> (N, 6, 50, H)
                batch_class_tokens = self.img_model.class_token.expand(merged_embedding.shape[1], -1, -1).repeat(N, 1, 1, 1)
                merged_embedding = torch.cat([batch_class_tokens, merged_embedding], dim=2)

            # Add positional embeddings and remove class token -> (N, 6, 49, H)
                merged_embedding += self.img_model.encoder.pos_embedding.repeat(N, 1, 1, 1)
                merged_embedding = merged_embedding[:, :, 1:]

            # Get merged embedding and reshape to 2D embedding -> (N, 1, 49, H)
                merged_embedding = torch.stack([self.gpa(embedding.flatten(start_dim=1)).reshape(VIT_SEQ_LENGTH,
                                                                                                VIT_HIDDEN_STATE) for
                                                embedding in merged_embedding], dim=0)

                # Project to VL dimension -> (1, 49, H) (H is 512 for t5-small, 768 for t5-base)
                if self.lm != 'T5-Base':
                    merged_embedding = self.img_projection_layer(merged_embedding)
                # if self.lm not in ['T5-Base', 'T5-Large']:
                #     merged_embedding = self.img_projection_layer(merged_embedding)


                # Add modal type embedding to merged embedding
                merged_embedding += self.modal_embeddings(
                    torch.ones((1, merged_embedding.shape[1]), dtype=torch.int, device=device))

            return merged_embedding
        
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
                pose_features = self.pose_encoder(flattened_pose)
                
                # Reshape for attention: [batch_size, 1, hidden_size]
                pose_features = pose_features.unsqueeze(1)
                
                # Apply self-attention to keypoints
                attended_pose, _ = self.keypoint_attention(pose_features, pose_features, pose_features)
                
                # Final projection
                output = self.final_projection(attended_pose.squeeze(1))  # [batch_size, hidden_size]
                
                return output
        

        def forward(self, text_enc, imgs, text_model):
           
            # Get the image embeddings (N x 1 x 49 x H)
            imgs_embedding = self.get_img_embedding(imgs)

           

            # # Get the text embeddings (N x S x H)
            text_embeddings = text_model.get_input_embeddings()(text_enc)

            # Add modal embeddings to text
            text_embeddings += self.modal_embeddings(torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int,
                                                                 device=device))
            if self.attention:
                imgs_embedding,text_embeddings=self.atten( imgs_embedding, text_embeddings)
            # Concatenate embeddings -> (1 x S x 512)
            merged_embedding = torch.cat([text_embeddings, imgs_embedding], dim=1)

            return merged_embedding

    def forward(self, text_enc, imgs, labels=None, bodypose=None, local_content_features=None, local_motion_features=None):

        # Get the merged embeddings
        merged_embedding = self.mvp(text_enc, imgs, self.model)
        
        # Process body pose if available
        if bodypose is not None and bodypose.numel() > 0:
            pose_features = self.bodypose_processor(bodypose)
            # Add pose features to the merged embedding
            # We'll add it to the image part of the embedding
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                # Add pose features to the image part
                pose_embedding = pose_features.unsqueeze(1).expand(-1, 49, -1)  # Expand to match image sequence length
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + pose_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                # If only image embeddings, add pose features directly
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
                # Create projection layer if it doesn't exist
                self.local_features_projection = nn.Linear(256, merged_embedding.shape[-1]).to(device)
                local_features = self.local_features_projection(local_features)
            
            # Add local features to the merged embedding
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                # Add local features to the image part
                local_embedding = local_features.unsqueeze(1).expand(-1, 49, -1)
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + local_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                # If only image embeddings, add local features directly
                local_embedding = local_features.unsqueeze(1).expand(-1, merged_embedding.shape[1], -1)
                merged_embedding = merged_embedding + local_embedding

        # If training include the labels
     
        out_t5=self.model(inputs_embeds=merged_embedding, labels=labels)
        int_output=self.int(out_t5.logits)
        return out_t5,int_output

    def generate(self, text_enc, imgs, bodypose=None, lidar=None, local_content_features=None, local_motion_features=None):
        # self.model.eval()
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
                # Create projection layer if it doesn't exist
                self.local_features_projection = nn.Linear(256, merged_embedding.shape[-1]).to(device)
                local_features = self.local_features_projection(local_features)
            
            # Add local features to the merged embedding
            if merged_embedding.shape[1] > 49:  # If we have text + image embeddings
                # Add local features to the image part
                local_embedding = local_features.unsqueeze(1).expand(-1, 49, -1)
                merged_embedding = torch.cat([merged_embedding[:, :49, :] + local_embedding, merged_embedding[:, 49:, :]], dim=1)
            else:
                # If only image embeddings, add local features directly
                local_embedding = local_features.unsqueeze(1).expand(-1, merged_embedding.shape[1], -1)
                merged_embedding = merged_embedding + local_embedding

        attention_mask = torch.ones(merged_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((merged_embedding.shape[0], 1), dtype=torch.long, device=device)*self.model.config.decoder_start_token_id
        output_ids = self.model.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=merged_embedding, max_length=512, early_stopping=True)
        # out_t5=self.model(inputs_embeds=merged_embedding, decoder_input_ids=decoder_input_ids)
        # int_out=self.int(out_t5.logits)

        return output_ids # int_out
