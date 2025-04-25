import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import json
import random
from pathlib import Path

# Constants matching the Phantom codebase
TRAJECTORY_SIZE = 33
DESIRE_LEN = 8
TRAFFIC_CONVENTION_LEN = 2
PLAN_MHP_N = 5
PLAN_MHP_COLUMNS = 15
LEAD_MHP_N = 2
LEAD_TRAJ_LEN = 6
LEAD_PRED_DIM = 4
STOP_LINE_MHP_N = 3
STOP_LINE_PRED_DIM = 8
META_STRIDE = 7
NUM_META_INTERVALS = 5

# Output indices in array - must match Phantom exactly
PLAN_MHP_GROUP_SIZE = (2*PLAN_MHP_COLUMNS*TRAJECTORY_SIZE + 1)
LEAD_MHP_GROUP_SIZE = (2*LEAD_PRED_DIM*LEAD_TRAJ_LEN + 3)
STOP_LINE_MHP_GROUP_SIZE = (2*STOP_LINE_PRED_DIM + 1)

PLAN_IDX = 0
LL_IDX = PLAN_IDX + PLAN_MHP_N*PLAN_MHP_GROUP_SIZE
LL_PROB_IDX = LL_IDX + 4*2*2*TRAJECTORY_SIZE
RE_IDX = LL_PROB_IDX + 8
LEAD_IDX = RE_IDX + 2*2*2*TRAJECTORY_SIZE
LEAD_PROB_IDX = LEAD_IDX + LEAD_MHP_N*(LEAD_MHP_GROUP_SIZE)
STOP_LINE_IDX = LEAD_PROB_IDX + 3
STOP_LINE_PROB_IDX = STOP_LINE_IDX + STOP_LINE_MHP_N*STOP_LINE_MHP_GROUP_SIZE
DESIRE_STATE_IDX = STOP_LINE_PROB_IDX + 1
META_IDX = DESIRE_STATE_IDX + DESIRE_LEN
OTHER_META_SIZE = 48
DESIRE_PRED_SIZE = 32
POSE_IDX = META_IDX + OTHER_META_SIZE + DESIRE_PRED_SIZE
POSE_SIZE = 12
OUTPUT_SIZE = POSE_IDX + POSE_SIZE
TEMPORAL_SIZE = 512  # Temporal feature size

# Model input dimensions - must match commonmodel.h
MODEL_WIDTH = 512
MODEL_HEIGHT = 256

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=0):
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

class SuperComboNet(nn.Module):
    def __init__(self, temporal_size=TEMPORAL_SIZE):
        super(SuperComboNet, self).__init__()
        
        # Base CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(32, 64, 3, stride=2),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
        )
        
        # Recurrent component
        self.gru = nn.GRU(512, temporal_size, batch_first=True)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + temporal_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # Single output head producing the full output array
        self.output_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, OUTPUT_SIZE)
        )
        
        # Initialize temporal feature
        self.temporal_feature = torch.zeros(1, 1, temporal_size)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First block with downsampling
        layers.append(self._make_block(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        if downsample is not None:
            return nn.Sequential(*layers, downsample, nn.ReLU(inplace=True))
        else:
            return nn.Sequential(*layers, nn.ReLU(inplace=True))
    
    def reset_temporal_state(self, batch_size=1):
        self.temporal_feature = torch.zeros(batch_size, 1, self.gru.hidden_size, device=next(self.parameters()).device)
    
    def forward(self, image, desire=None, traffic_convention=None, recurrent_state=None):
        batch_size = image.size(0)
        
        # CNN feature extraction
        x = self.encoder(image)
        x = torch.mean(x, dim=(2, 3))  # Global average pooling
        
        # Process temporal features
        if recurrent_state is not None:
            self.temporal_feature = recurrent_state
        elif self.temporal_feature.size(0) != batch_size:
            self.reset_temporal_state(batch_size)
        
        # Process recurrent component
        x_gru = x.unsqueeze(1)  # Add sequence dimension
        gru_out, h_n = self.gru(x_gru, self.temporal_feature.transpose(0, 1).contiguous())
        self.temporal_feature = h_n.transpose(0, 1)
        
        # Feature fusion
        combined = torch.cat([x, gru_out.squeeze(1)], dim=1)
        fused = self.fusion(combined)
        
        # Generate flat output array
        output = self.output_head(fused)
        
        # In the main inference mode, we only return the primary output
        if not self.training:
            return output, self.temporal_feature
        
        # For training purposes, we can also return separated outputs for easier loss calculation
        output_dict = {
            'full_output': output,
            'path': output[:, PLAN_IDX:LL_IDX],
            'lane_lines': output[:, LL_IDX:RE_IDX],
            'road_edges': output[:, RE_IDX:LEAD_IDX],
            'lead': output[:, LEAD_IDX:STOP_LINE_IDX],
            'stop_line': output[:, STOP_LINE_IDX:DESIRE_STATE_IDX],
            'meta': output[:, DESIRE_STATE_IDX:POSE_IDX],
            'pose': output[:, POSE_IDX:POSE_IDX+POSE_SIZE],
            'temporal': self.temporal_feature
        }
        
        return output_dict

# RGB to YUV conversion
def rgb_to_yuv(rgb_image):
    """Convert RGB image tensor to YUV format"""
    # RGB to YUV conversion matrix
    transform = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], dtype=torch.float32, device=rgb_image.device)
    
    # Reshape for batch matrix multiplication
    batch_size, channels, height, width = rgb_image.shape
    reshaped_img = rgb_image.permute(0, 2, 3, 1).reshape(batch_size, height*width, 3)
    
    # Convert to YUV
    yuv = torch.bmm(reshaped_img, transform.T.repeat(batch_size, 1, 1))
    
    # Reshape back
    return yuv.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)

class DrivingDataset(Dataset):
    def __init__(self, data_dir, transform=None, limit_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all image files
        self.image_files = list(self.data_dir.glob('**/images/*.jpg')) + list(self.data_dir.glob('**/images/*.png'))
        print(f"Found {len(self.image_files)} image files")
        
        # Limit dataset size for debugging
        if limit_samples:
            self.image_files = self.image_files[:limit_samples]
        
        # For each image, there should be a corresponding label file
        self.label_files = []
        for img_file in self.image_files:
            # Assuming labels are in a parallel directory structure with the same filename but .json extension
            label_file = img_file.parent.parent / 'labels' / f"{img_file.stem}.json"
            if label_file.exists():
                self.label_files.append(label_file)
            else:
                print(f"Warning: No label file found for {img_file}")
        
        # Keep only images with labels
        self.image_files = [self.image_files[i] for i in range(len(self.image_files)) if i < len(self.label_files)]
        
        print(f"Using {len(self.image_files)} images with valid labels")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize image to expected input size
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        label_path = self.label_files[idx]
        with open(label_path, 'r') as f:
            labels = json.load(f)
        
        # Initialize a full output array with zeros
        output = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        
        # Process labels into the full output array format
        if 'path' in labels:
            for i, path in enumerate(labels['path'][:PLAN_MHP_N]):
                path_offset = PLAN_IDX + i * PLAN_MHP_GROUP_SIZE
                for j, point in enumerate(path['points'][:TRAJECTORY_SIZE]):
                    # Position (x,y,z)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 0] = point['x']
                    output[path_offset + j*PLAN_MHP_COLUMNS + 1] = point['y']
                    output[path_offset + j*PLAN_MHP_COLUMNS + 2] = point['z']
                    
                    # Velocity (vx,vy,vz)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 3] = point.get('vx', 0.0)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 4] = point.get('vy', 0.0)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 5] = point.get('vz', 0.0)
                    
                    # Orientation (roll,pitch,yaw)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 9] = point.get('roll', 0.0)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 10] = point.get('pitch', 0.0)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 11] = point.get('yaw', 0.0)
                    
                    # Orientation rate (roll_rate,pitch_rate,yaw_rate)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 12] = point.get('roll_rate', 0.0)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 13] = point.get('pitch_rate', 0.0)
                    output[path_offset + j*PLAN_MHP_COLUMNS + 14] = point.get('yaw_rate', 0.0)
                    
                    # Standard deviations - stored as log values
                    if 'std' in point:
                        std_offset = path_offset + TRAJECTORY_SIZE * PLAN_MHP_COLUMNS
                        output[std_offset + j*PLAN_MHP_COLUMNS + 0] = np.log(point['std'].get('x', 1.0))
                        output[std_offset + j*PLAN_MHP_COLUMNS + 1] = np.log(point['std'].get('y', 1.0))
                        output[std_offset + j*PLAN_MHP_COLUMNS + 2] = np.log(point['std'].get('z', 1.0))
                        # ... same for velocity and orientation
        
        # Lane lines
        if 'lane_lines' in labels:
            for i, lane in enumerate(labels['lane_lines'][:4]):
                lane_offset = LL_IDX + i * 2 * 2 * TRAJECTORY_SIZE
                for j, point in enumerate(lane['points'][:TRAJECTORY_SIZE]):
                    output[lane_offset + j*2 + 0] = point['x']
                    output[lane_offset + j*2 + 1] = point['y']
                    
                    # Standard deviations stored as separate arrays
                    std_offset = lane_offset + 4 * TRAJECTORY_SIZE
                    if 'std' in point:
                        output[std_offset + j*2 + 0] = np.log(point['std'].get('x', 1.0))
                        output[std_offset + j*2 + 1] = np.log(point['std'].get('y', 1.0))
                
                # Lane line probabilities (stored as logits for sigmoid)
                prob_offset = LL_PROB_IDX + i * 2
                output[prob_offset + 1] = np.log(lane.get('probability', 0.5) / (1 - lane.get('probability', 0.5) + 1e-6))
        
        # Similarly for road edges, leads, stop lines...
        # ... (detailed implementation for each component)
        
        # Convert processed data to tensor
        target = torch.from_numpy(output)
        
        # Traffic convention & desire inputs
        desire_input = np.zeros(DESIRE_LEN, dtype=np.float32)
        if 'desire_input' in labels:
            desire_idx = labels['desire_input']
            if 0 <= desire_idx < DESIRE_LEN:
                desire_input[desire_idx] = 1.0
        else:
            desire_input[0] = 1.0  # Default to 'none'
            
        traffic_convention = np.zeros(TRAFFIC_CONVENTION_LEN, dtype=np.float32)
        if 'traffic_convention' in labels:
            is_rhd = labels['traffic_convention'].get('is_right_hand_drive', False)
            traffic_convention[1 if is_rhd else 0] = 1.0
        else:
            traffic_convention[0] = 1.0  # Default to left-hand drive
            
        return {
            'image': image,
            'target': target,
            'desire': torch.from_numpy(desire_input),
            'traffic_convention': torch.from_numpy(traffic_convention)
        }

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
            outputs = model(images, desire, traffic_convention)
            
            # Calculate loss on full output array
            loss = nn.MSELoss()(outputs['full_output'], targets)
            
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
                outputs = model(images, desire, traffic_convention)
                
                # Calculate validation loss
                loss = nn.MSELoss()(outputs['full_output'], targets)
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
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples (for debugging)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define YUV transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # We'll convert to YUV in the forward pass
    ])
    
    # Create dataset
    full_dataset = DrivingDataset(
        data_dir=args.data_dir,
        transform=transform,
        limit_samples=args.limit_samples
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = SuperComboNet(temporal_size=TEMPORAL_SIZE)
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
