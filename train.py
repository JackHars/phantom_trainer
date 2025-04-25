import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from constants import *
from models import SuperComboNet
from datasets import CityscapesDataset, Comma10kDataset

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            desire = batch['desire'].to(device)
            traffic_convention = batch['traffic_convention'].to(device)
            
            # Forward pass
            outputs, _ = model(images, desire, traffic_convention)
            
            # Calculate loss on full output array
            loss = nn.MSELoss()(outputs, targets)
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch['image'].to(device)
                targets = batch['target'].to(device)
                desire = batch['desire'].to(device)
                traffic_convention = batch['traffic_convention'].to(device)
                
                # Reset temporal state for consistency
                model.reset_temporal_state(images.size(0))
                
                # Forward pass
                outputs, _ = model(images, desire, traffic_convention)
                
                # Calculate validation loss
                loss = nn.MSELoss()(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'supercombo_best.pth')
            
            # Export to ONNX format in a way compatible with Phantom
            dummy_image = torch.randn(1, 3, MODEL_HEIGHT, MODEL_WIDTH, device=device)
            
            # Set model to inference mode
            model.eval()
            
            # Export the model to ONNX
            torch.onnx.export(
                model,
                dummy_image,  # Just pass the image as input
                "supercombo.onnx",
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],  # Single input name
                output_names=['output', 'recurrent'],  # Main output and recurrent state
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                    'recurrent': {0: 'batch_size'}
                }
            )
            print(f"Saved best model at epoch {epoch+1} with validation loss {val_loss:.6f}")
            print(f"Exported ONNX model to supercombo.onnx")

def main():
    parser = argparse.ArgumentParser(description='Train SuperCombo Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing dataset')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'comma10k'], 
                      help='Dataset to use for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples (for debugging)')
    parser.add_argument('--annotation_mode', type=str, default='fine', choices=['fine', 'coarse'], 
                      help='Annotation mode for Cityscapes dataset')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets based on the chosen dataset
    if args.dataset == 'cityscapes':
        print(f"Using Cityscapes dataset from {args.data_dir}")
        train_dataset = CityscapesDataset(
            data_dir=args.data_dir,
            transform=transform,
            split='train',
            mode=args.annotation_mode,
            limit_samples=args.limit_samples
        )
        
        val_dataset = CityscapesDataset(
            data_dir=args.data_dir,
            transform=transform,
            split='val',
            mode=args.annotation_mode,
            limit_samples=args.limit_samples
        )
    elif args.dataset == 'comma10k':
        print(f"Using comma10k dataset from {args.data_dir}")
        train_dataset = Comma10kDataset(
            data_dir=args.data_dir,
            transform=transform,
            split='train',
            limit_samples=args.limit_samples
        )
        
        val_dataset = Comma10kDataset(
            data_dir=args.data_dir,
            transform=transform,
            split='val',
            limit_samples=args.limit_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = SuperComboNet(temporal_size=TEMPORAL_SIZE, output_size=OUTPUT_SIZE)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate
    )

if __name__ == "__main__":
    main() 