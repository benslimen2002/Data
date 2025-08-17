from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms
from .local_features import PedestrianCropper
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PedDataset(Dataset):

    def __init__(self, input_file, config, tokenizer=None, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.config = config
        self.transform = transform
        self.bodypose = True  # pour activer/désactiver
        self.bodypose_path = "data/JAAD/bodypose_yolo/"
        self.use_bodypose = True  # pour activer/désactiver l'utilisation des données de pose
        
        # Local features configuration
        self.use_local_features = getattr(config, 'use_local_features', True)
        self.sequence_length = getattr(config, 'sequence_length', 5)  # m frames in the paper
        self.pedestrian_cropper = PedestrianCropper(target_size=(224, 224))

        # Initialize tokenizer if not provided
        if tokenizer is None:
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        else:
            self.tokenizer = tokenizer

        # Initialize transform if not provided
        if self.transform is None:
            from torchvision import transforms
            # Create a transform that works with tensors, not PIL images
            self.transform = transforms.Compose([
                # No Resize since it expects PIL images
                # No ToTensor since read_image already returns tensors
                # No Normalize to avoid range issues
            ])

        if self.config.encoder=='clip':
            self.precossor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # Get the question and answer at the idx
     qa, img_path = self.data[idx]
     img_path = list(img_path.values())

     q_text, a_text, i_label = qa['Q'], qa['A'], qa['C']
     q_text = f"Question: {q_text} Answer:"
     
     # Extract bounding box coordinates from the question text
     bbox_coords = self._extract_bbox_from_question(q_text)

    # ======== IMAGES ========
     if self.config.encoder == 'clip':
        rgb_image = Image.open(img_path[0])
        rgb_image = self.precossor(images=rgb_image, return_tensors="pt")
        rgb_image = {k: v.squeeze(0) for k, v in rgb_image.items()}
        e_im = rgb_image['pixel_values'].to(device)

        if self.config.optical:
            opticalflow = Image.open(os.path.join(img_path[1])).convert('RGB')
            opticalflow = self.precossor(images=opticalflow, return_tensors="pt")
            opticalflow = {k: v.squeeze(0) for k, v in opticalflow.items()}
            op = opticalflow['pixel_values'].to(device)
            imgs = [e_im, op]
        else:
            imgs = [e_im]
     else:
        # Load RGB image and ensure proper normalization
        rgb_tensor = read_image(img_path[0]).float()
        rgb_tensor = rgb_tensor / 255.0  # Normalize to [0, 1]
        rgb_tensor = torch.clamp(rgb_tensor, 0, 1)  # Ensure values are in range
        
        # Resize tensor manually to 224x224
        rgb_tensor = F.interpolate(rgb_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        rgb_image = rgb_tensor.to(device)
        
        if self.config.optical:
            opticalflow = read_image(os.path.join(img_path[1]))
            if opticalflow.shape[0] > 3:
                new_channel = (opticalflow[2, :, :] + opticalflow[3, :, :]) / 2.0
            else:
                new_channel = opticalflow[2, :, :]
            opticalflow = torch.stack((opticalflow[0, :, :], opticalflow[1, :, :], new_channel))
            opticalflow = opticalflow.float() / 255.0
            opticalflow = torch.clamp(opticalflow, 0, 1)  # Ensure values are in range
            
            # Resize tensor manually to 224x224
            opticalflow = F.interpolate(opticalflow.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            
            opticalflow = opticalflow.to(device)
                
            imgs = [rgb_image, opticalflow]
        else:
            imgs = [rgb_image]

     imgs = torch.stack(imgs, dim=0)

     # ======== LOCAL FEATURES ========
     local_content_features = None
     local_motion_features = None
     
     if self.use_local_features and bbox_coords is not None:
         try:
             # Extract local features for the current frame
             local_content, local_motion = self._extract_local_features(
                 img_path[0], img_path[1], bbox_coords
             )
             local_content_features = local_content
             local_motion_features = local_motion
         except Exception as e:
             print(f"Warning: Could not extract local features: {e}")
             # Create proper tensor shapes for local features
             local_content_features = torch.zeros((1, 128), dtype=torch.float32).to(device)
             local_motion_features = torch.zeros((1, 128), dtype=torch.float32).to(device)
     else:
         # Create proper tensor shapes for local features
         local_content_features = torch.zeros((1, 128), dtype=torch.float32).to(device)
         local_motion_features = torch.zeros((1, 128), dtype=torch.float32).to(device)

         # ======== BODYPose ========
     if self.use_bodypose:
        try:
            # img_path[0] is RGB image path relative to img_path dir
            # Replace extension .jpg/.png with .npy and point to bodypose_dir
            frame_name = os.path.splitext(os.path.basename(img_path[0]))[0] + '.npy'
            video_name = os.path.basename(os.path.dirname(img_path[0]))
            bodypose_path = os.path.join(self.config.bodypose_path, video_name, frame_name)
            
            if os.path.exists(bodypose_path):
                bodypose_data = np.load(bodypose_path)  # shape (1, 17, 2)
                # Normalize coordinates to [0, 1] range
                bodypose_data = bodypose_data.reshape(-1, 2)  # reshape to (17, 2)
                # Normalize to [0, 1] range assuming image dimensions
                bodypose_data[:, 0] = bodypose_data[:, 0] / 1920.0  # normalize x coordinates
                bodypose_data[:, 1] = bodypose_data[:, 1] / 1080.0  # normalize y coordinates
                bodypose = torch.tensor(bodypose_data, dtype=torch.float32).to(device)
            else:
                # If bodypose file doesn't exist, create zero tensor
                bodypose = torch.zeros((17, 2), dtype=torch.float32).to(device)
        except Exception as e:
            print(f"Warning: Could not load bodypose data: {e}")
            bodypose = torch.zeros((17, 2), dtype=torch.float32).to(device)
     else:
        bodypose = torch.zeros((17, 2), dtype=torch.float32).to(device)  # if no bodypose, return zero tensor

     i_label = torch.tensor(i_label).to(device)

     return q_text, imgs, a_text, i_label, sorted(list(img_path)), bodypose, local_content_features, local_motion_features

    
    # def __getitem__(self, idx):
    #     # Get the question and answer at the idx
    #     qa, img_path = self.data[idx]
    #     img_path = list(img_path.values())

    #     q_text, a_text ,i_label= qa['Q'], qa['A'],qa['C']
    #     q_text = f"Question: {q_text} Answer:"

    #     #Concatenate images into a single tensor
    #     if self.config.encoder=='clip':
    #         opticalflow=Image.open(os.path.join(self.config.img_path,img_path[1])).convert('RGB')
    #         opticalflow=self.precossor(images=opticalflow,return_tensors="pt")
    #         opticalflow={k: v.squeeze(0) for k, v in opticalflow.items()}
    #         op=opticalflow['pixel_values'].to(device) 
            

    #         # resize_transform = transforms.Resize((224, 224))
    #         # rgb_image=Image.open(os.path.join(self.config.img_path,img_path[0]))
    #         # rgb_image=self.precossor(images=rgb_image,return_tensors="pt")
    #         # rgb_image={k: v.squeeze(0) for k, v in rgb_image.items()}
    #         # e_im=rgb_image['pixel_values'].to(device) 
    #         # rgb_image=resize_transform(rgb_image)
    #         if self.config.optical:
    #             # opticalflow=Image.open(os.path.join(self.config.img_path,img_path[1])).convert('RGB')
    #             # opticalflow=self.precossor(images=opticalflow,return_tensors="pt")
    #             # opticalflow={k: v.squeeze(0) for k, v in opticalflow.items()}
    #             # op=opticalflow['pixel_values'].to(device) 

    #             rgb_image=Image.open(os.path.join(self.config.img_path,img_path[0]))
    #             rgb_image=self.precossor(images=rgb_image,return_tensors="pt")
    #             rgb_image={k: v.squeeze(0) for k, v in rgb_image.items()}
    #             e_im=rgb_image['pixel_values'].to(device) 
    #             imgs=[e_im,op]
    #         else:
    #             # imgs=[e_im]
    #             imgs=[op]
    #     else:
        
    #         rgb_image=self.transform(read_image(os.path.join(self.config.img_path,img_path[0])).float()).to(device)
    #         if self.config.optical:
    #             opticalflow=read_image(os.path.join(self.config.img_path,img_path[1]))
    #             new_channel = (opticalflow[2, :, :] + opticalflow[3, :, :]) / 2.0
    #             opticalflow=self.transform(torch.stack((opticalflow[0, :, :], opticalflow[1, :, :], new_channel))).float().to(device) 
    #             imgs=[rgb_image,opticalflow]
    #         else:
    #             imgs=[rgb_image]
        
    #     imgs = torch.stack(imgs, dim=0)


    #     # imgs = [self.transform(os.path.join(self.config.data_path, img_path(p)).float()).to(device) for p in img_path]
    #     # imgs=[self.transform(read_image((os.path.join(self.config.data_path,p))).float()).to(device) for p in img_path]
    #     # [6,3,224,224]
            
    #     i_label=torch.tensor(i_label).to(device)

    #     return q_text, imgs, a_text, i_label,sorted(list(img_path))


    def collate_fn(self, batch):
        q_texts, imgs, a_texts, i_label, _, bodyposes, local_content_features, local_motion_features = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        i_label = torch.stack(list(i_label), dim=0)
        bodyposes = torch.stack(list(bodyposes), dim=0)
        local_content_features = torch.stack(list(local_content_features), dim=0)
        local_motion_features = torch.stack(list(local_motion_features), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)
        

        return encodings, imgs, labels, i_label, bodyposes, local_content_features, local_motion_features

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, i_labels, img_path, bodyposes, local_content_features, local_motion_features = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        i_labels = torch.stack(list(i_labels), dim=0)
        bodyposes = torch.stack(list(bodyposes), dim=0)
        local_content_features = torch.stack(list(local_content_features), dim=0)
        local_motion_features = torch.stack(list(local_motion_features), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)
       

        return list(q_texts), encodings, imgs, labels, i_labels, img_path, bodyposes, local_content_features, local_motion_features

    def _extract_bbox_from_question(self, question_text):
        """
        Extract bounding box coordinates from the question text
        The format is: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
        try:
            # Look for bounding box pattern in the question
            import re
            bbox_pattern = r'\[\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\](?:,\s*\[[\d\.]+,\s*[\d\.]+,\s*[\d\.]+,\s*[\d\.]+\])*\]'
            match = re.search(bbox_pattern, question_text)
            
            if match:
                # Extract the first bounding box (most recent)
                bbox_str = match.group(0)
                # Parse the first set of coordinates
                first_bbox = re.findall(r'\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]', bbox_str)[0]
                bbox = [float(x) for x in first_bbox]
                return bbox
            
            return None
        except Exception as e:
            print(f"Warning: Could not extract bounding box: {e}")
            return None

    def _extract_local_features(self, rgb_path, optical_flow_path, bbox):
        """
        Extract local content and motion features for a single frame
        
        Args:
            rgb_path: Path to RGB image
            optical_flow_path: Path to optical flow image
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            local_content: Local content features
            local_motion: Local motion features
        """
        # Load RGB image - img_path already contains full path
        rgb_image = Image.open(rgb_path)
        
        # Load optical flow image - img_path already contains full path
        optical_flow = Image.open(optical_flow_path)
        
        # Crop pedestrian region
        cropped_rgb = self.pedestrian_cropper.crop_and_warp(rgb_image, bbox)
        cropped_flow = self.pedestrian_cropper.crop_and_warp(optical_flow, bbox)
        
        # Convert to tensors - handle PIL images properly
        if isinstance(cropped_rgb, Image.Image):
            # Convert PIL to tensor manually
            rgb_tensor = torch.tensor(np.array(cropped_rgb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        else:
            rgb_tensor = self.transform(cropped_rgb).unsqueeze(0) if self.transform else torch.tensor(np.array(cropped_rgb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
        if isinstance(cropped_flow, Image.Image):
            # Convert PIL to tensor manually
            flow_tensor = torch.tensor(np.array(cropped_flow)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        else:
            flow_tensor = self.transform(cropped_flow).unsqueeze(0) if self.transform else torch.tensor(np.array(cropped_flow)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Ensure tensors are in [0, 1] range
        rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
        flow_tensor = torch.clamp(flow_tensor, 0, 1)
        
        # Simple feature extraction (placeholder)
        local_content = torch.mean(rgb_tensor, dim=[2, 3])  # [1, 3]
        local_motion = torch.mean(flow_tensor, dim=[2, 3])  # [1, 3]
        
        # Project to 128 dimensions to match the expected output
        local_content = F.linear(local_content, torch.randn(128, 3).to(device))  # [1, 128]
        local_motion = F.linear(local_motion, torch.randn(128, 3).to(device))   # [1, 128]
        
        return local_content, local_motion
