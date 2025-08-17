#!/usr/bin/env python3
"""
Script d'entraînement complet pour le modèle PedVLM avec 6 entrées
- RGB Images
- Optical Flow  
- Text
- Body Pose
- Local Content (Plc) - VGG19 + GRU sur images pédestres croppées
- Local Motion (Plm) - Conv3D + MP3D + GRU sur optical flow pédestre

Basé sur train.py original avec ajout des local features
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Import du modèle avec local features
from modules.Ped_model_simple import PedVLMT5
from modules.Ped_dataset import PedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_config(config_path):
    """Charge la configuration depuis le fichier JSON"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Crée un objet simple pour contenir les valeurs de config
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            
            # Ajoute les attributs manquants s'ils n'existent pas
            if not hasattr(self, 'optical'):
                self.optical = 'data/JAAD/optical_flow'
            if not hasattr(self, 'num_workers'):
                self.num_workers = 0
            if not hasattr(self, 'result_path'):
                self.result_path = 'results'
            if not hasattr(self, 'checkpoint_frequency'):
                self.checkpoint_frequency = 5
            if not hasattr(self, 'weight_decay'):
                self.weight_decay = 0.01
            if not hasattr(self, 'use_local_features'):
                self.use_local_features = True
            if not hasattr(self, 'sequence_length'):
                self.sequence_length = 5
    
    return Config(**config)

def create_datasets(config):
    """Crée les datasets d'entraînement, validation et test"""
    print("Creating datasets...")
    
    # Crée le transform compatible (sans ToTensor car images déjà tensors)
    from torchvision import transforms
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Pas de ToTensor car images déjà tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crée les datasets avec les bons paramètres
    train_dataset = PedDataset(
        input_file='Ped_Dataset/train.json',
        config=config,
        tokenizer=None,  # Géré en interne
        transform=default_transform
    )
    
    val_dataset = PedDataset(
        input_file='Ped_Dataset/val.json',
        config=config,
        tokenizer=None,
        transform=default_transform
    )
    
    test_dataset = PedDataset(
        input_file='Ped_Dataset/test.json',
        config=config,
        tokenizer=None,
        transform=default_transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def create_model(config):
    """Crée et retourne le modèle"""
    print("Creating model...")
    model = PedVLMT5(config).to(device)
    
    # Compte les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Entraîne pour une époque"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Dépaquette le batch - gère les batches de 7 et 5 éléments
            if len(batch) == 7:
                # Avec local features
                text_enc, imgs, labels, i_labels, bodyposes, local_content_features, local_motion_features = batch
            else:
                # Sans local features
                text_enc, imgs, labels, i_labels, bodyposes = batch
                local_content_features = None
                local_motion_features = None
            
            # Déplace vers le device
            text_enc = text_enc.to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)
            i_labels = i_labels.to(device)
            bodyposes = bodyposes.to(device)
            
            if local_content_features is not None:
                local_content_features = local_content_features.to(device)
                local_motion_features = local_motion_features.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, int_outputs = model(
                text_enc, imgs, labels, bodyposes, 
                local_content_features, local_motion_features
            )
            
            # Calcule les pertes
            main_loss = outputs.loss
            int_loss = criterion(int_outputs, i_labels)
            total_loss_batch = main_loss + int_loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            # Collecte les prédictions et labels pour les métriques
            predictions = torch.argmax(int_outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(i_labels.cpu().numpy())
            
            total_loss += total_loss_batch.item()
            
            # Met à jour la barre de progression
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Main Loss': f'{main_loss.item():.4f}',
                'Int Loss': f'{int_loss.item():.4f}'
            })
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    # Calcule les métriques
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return total_loss / len(train_loader), accuracy, f1, precision, recall

def evaluate_model(model, val_loader, criterion):
    """Évalue le modèle sur l'ensemble de validation"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Dépaquette le batch - gère les batches de 7 et 5 éléments
                if len(batch) == 7:
                    # Avec local features
                    text_enc, imgs, labels, i_labels, bodyposes, local_content_features, local_motion_features = batch
                else:
                    # Sans local features
                    text_enc, imgs, labels, i_labels, bodyposes = batch
                    local_content_features = None
                    local_motion_features = None
                
                # Déplace vers le device
                text_enc = text_enc.to(device)
                imgs = imgs.to(device)
                labels = labels.to(device)
                i_labels = i_labels.to(device)
                bodyposes = bodyposes.to(device)
                
                if local_content_features is not None:
                    local_content_features = local_content_features.to(device)
                    local_motion_features = local_motion_features.to(device)
                
                # Forward pass
                outputs, int_outputs = model(
                    text_enc, imgs, labels, bodyposes, 
                    local_content_features, local_motion_features
                )
                
                # Calcule les pertes
                main_loss = outputs.loss
                int_loss = criterion(int_outputs, i_labels)
                total_loss_batch = main_loss + int_loss
                
                # Collecte les prédictions et probabilités
                probs = torch.softmax(int_outputs, dim=1)
                predictions = torch.argmax(int_outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(i_labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilité de la classe positive
                
                total_loss += total_loss_batch.item()
                
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
    
    # Calcule les métriques
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Calcule l'AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return (total_loss / len(val_loader), accuracy, f1, precision, recall, auc)

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    """Trace les courbes d'entraînement"""
    epochs = range(1, len(train_losses) + 1)
    
    # Trace les pertes
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Trace les métriques
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
    for i, metric in enumerate(metrics):
        plt.plot(epochs, [m[i] for m in train_metrics], 'b-', label=f'Train {metric}')
        plt.plot(epochs, [m[i] for m in val_metrics], 'r-', label=f'Val {metric}')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def save_results(results, save_dir):
    """Sauvegarde les résultats d'entraînement"""
    # Sauvegarde les métriques
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Sauvegarde le modèle final
    torch.save(results['final_model_state'], os.path.join(save_dir, 'final_model.pth'))
    
    print(f"Results saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train PedVLM with 6 inputs including Local Content and Local Motion')
    parser.add_argument('--config', default='config_local_features.json', help='Path to config file')
    parser.add_argument('--data_path', default='Ped_Dataset', help='Path to dataset')
    parser.add_argument('--bodypose_path', default='data/JAAD/bodypose_yolo', help='Path to bodypose data')
    parser.add_argument('--img_path', default='data/JAAD/images', help='Path to images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Charge la configuration
    config = load_config(args.config)
    
    # Remplace la config avec les arguments de ligne de commande
    config.data_path = args.data_path
    config.bodypose_path = args.bodypose_path
    config.img_path = args.img_path
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    
    # Crée le répertoire de sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(config.result_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Crée les datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Crée les data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
        num_workers=config.num_workers
    )
    
    # Crée le modèle
    model = create_model(config)
    
    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Scheduler de learning rate
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Boucle d'entraînement
    print("Starting training...")
    print("=" * 50)
    
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'=' * 50}")
        
        # Entraîne
        train_loss, train_acc, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        
        # Évalue
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = evaluate_model(
            model, val_loader, criterion
        )
        
        # Met à jour le learning rate
        scheduler.step()
        
        # Stocke les métriques
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append([train_acc, train_f1, train_prec, train_rec])
        val_metrics.append([val_acc, val_f1, val_prec, val_rec])
        
        # Affiche les résultats de l'époque
        print(f"\nEpoch {epoch} Results:")
        print(f"Training - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Sauvegarde le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_epoch_{epoch}.pth'))
            print(f"New best model saved! (Epoch {epoch})")
        
        # Sauvegarde le checkpoint
        if epoch % config.checkpoint_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Évaluation finale
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    final_val_loss, final_val_acc, final_val_f1, final_val_prec, final_val_rec, final_val_auc = evaluate_model(
        model, val_loader, criterion
    )
    
    print(f"Final Validation Results:")
    print(f"Loss: {final_val_loss:.4f}")
    print(f"Accuracy: {final_val_acc:.4f}")
    print(f"F1 Score: {final_val_f1:.4f}")
    print(f"Precision: {final_val_prec:.4f}")
    print(f"Recall: {final_val_rec:.4f}")
    print(f"AUC: {final_val_auc:.4f}")
    
    # Trace les courbes d'entraînement
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_dir)
    
    # Sauvegarde les résultats
    results = {
        'config': vars(config),
        'final_metrics': {
            'val_loss': final_val_loss,
            'val_accuracy': final_val_acc,
            'val_f1': final_val_f1,
            'val_precision': final_val_prec,
            'val_recall': final_val_rec,
            'val_auc': final_val_auc
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        },
        'final_model_state': model.state_dict()
    }
    
    save_results(results, save_dir)
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
