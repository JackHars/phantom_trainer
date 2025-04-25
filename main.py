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

# Cityscapes class definitions
CITYSCAPES_CLASSES = {
    0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle'
}

# Class groups
ROAD_CLASSES = [7, 8, 9, 10]  # road, sidewalk, parking, rail track
VEHICLE_CLASSES = [26, 27, 28, 29, 30, 31, 32, 33]  # car, truck, bus, etc.
HUMAN_CLASSES = [24, 25]  # person, rider
TRAFFIC_CONTROL_CLASSES = [19, 20]  # traffic light, traffic sign

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

class CityscapesDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', mode='fine', limit_samples=None):
        """
        Cityscapes dataset for driving model training
        
        Args:
            data_dir: Root directory of Cityscapes dataset
            transform: Optional transforms to apply
            split: 'train', 'val', or 'test'
            mode: 'fine' or 'coarse' annotations
            limit_samples: Optional limit to dataset size
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.mode = mode
        
        # Define paths based on Cityscapes structure
        self.images_dir = self.data_dir / 'leftImg8bit' / split
        self.annotations_dir = self.data_dir / f'gtFine_{mode}' / split
        
        # Find all image files
        self.image_files = list(self.images_dir.glob('**/*.png'))
        print(f"Found {len(self.image_files)} {split} images with {mode} annotations")
        
        # Limit dataset size for debugging
        if limit_samples:
            self.image_files = self.image_files[:limit_samples]
    
    def __len__(self):
        return len(self.image_files)
    
    def _get_annotation_path(self, img_path):
        """Get corresponding annotation path for an image"""
        city = img_path.parent.name
        img_name = img_path.stem.replace('_leftImg8bit', '')
        
        # Look for instance or semantic segmentation based on annotation type
        if self.mode == 'fine':
            # For fine annotations, we prefer instance segmentation
            instance_path = self.annotations_dir / city / f"{img_name}_gtFine_instanceIds.png"
            semantic_path = self.annotations_dir / city / f"{img_name}_gtFine_labelIds.png"
            
            if instance_path.exists():
                return instance_path, semantic_path
            elif semantic_path.exists():
                return None, semantic_path
        else:
            # For coarse annotations, we only have semantic segmentation
            semantic_path = self.annotations_dir / city / f"{img_name}_gtCoarse_labelIds.png"
            if semantic_path.exists():
                return None, semantic_path
            
        return None, None
    
    def _process_segmentation(self, seg_map, instance_map=None):
        """Process segmentation maps into our model's expected output format"""
        # Initialize a full output array with zeros
        output = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        
        # Extract road and lane information (for lane_lines and road_edges)
        # We'll use the road classes to identify the road edges and lane lines
        road_mask = np.isin(seg_map, ROAD_CLASSES)
        
        # Simple connected component analysis to find road region
        if np.any(road_mask):
            # Get road contours
            road_mask = road_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Use the largest contour as the road edge
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # Simplify the contour and sample points along it for road_edges
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Sample points evenly along the contour for road edges (left and right)
                n_points = min(TRAJECTORY_SIZE, len(approx_contour))
                indices = np.linspace(0, len(approx_contour) - 1, n_points).astype(int)
                
                # Left road edge
                for i, idx in enumerate(indices[:TRAJECTORY_SIZE]):
                    if i < len(indices):
                        point = approx_contour[idx][0]
                        # Normalize coordinates to range [-1, 1]
                        x = (point[0] / seg_map.shape[1] * 2) - 1
                        y = (point[1] / seg_map.shape[0] * 2) - 1
                        
                        # Store in road_edges section
                        edge_offset = RE_IDX + i * 2
                        output[edge_offset + 0] = x
                        output[edge_offset + 1] = y
        
        # For path planning, we'll use the center of the road as a simple default path
        if np.any(road_mask):
            # Find the center line of the road
            # This is a simplification - in real data you'd want to use actual path data
            skeleton = cv2.ximgproc.thinning(road_mask)
            if np.any(skeleton):
                # Get skeleton points
                y_indices, x_indices = np.where(skeleton > 0)
                
                # Sort points by y-coordinate (from bottom to top)
                sort_idx = np.argsort(y_indices)[::-1]
                y_sorted = y_indices[sort_idx]
                x_sorted = x_indices[sort_idx]
                
                # Sample evenly 
                n_points = min(TRAJECTORY_SIZE, len(y_sorted))
                if n_points > 0:
                    indices = np.linspace(0, len(y_sorted) - 1, n_points).astype(int)
                    
                    # Path 0 (main path)
                    path_offset = PLAN_IDX
                    
                    for i, idx in enumerate(indices):
                        if i < TRAJECTORY_SIZE:
                            # Normalize coordinates
                            x = (x_sorted[idx] / seg_map.shape[1] * 2) - 1
                            y = (y_sorted[idx] / seg_map.shape[0] * 2) - 1
                            
                            # Store in path section
                            point_offset = path_offset + i * PLAN_MHP_COLUMNS
                            output[point_offset + 0] = x  # x
                            output[point_offset + 1] = y  # y
                            output[point_offset + 2] = 0  # z
                            
                            # Velocity is forward direction (simplified)
                            if i < n_points - 1 and i + 1 < len(indices):
                                next_idx = indices[i + 1]
                                dx = (x_sorted[next_idx] - x_sorted[idx]) / seg_map.shape[1] * 2
                                dy = (y_sorted[next_idx] - y_sorted[idx]) / seg_map.shape[0] * 2
                                
                                # Simple velocity estimation
                                output[point_offset + 3] = dx * 10  # vx (scaled)
                                output[point_offset + 4] = dy * 10  # vy (scaled)
        
        # Find vehicle instances for lead vehicle prediction
        if instance_map is not None:
            vehicle_instance_ids = []
            
            # Check each vehicle class
            for vehicle_class in VEHICLE_CLASSES:
                # Find pixels with this class
                vehicle_pixels = (seg_map == vehicle_class)
                
                if np.any(vehicle_pixels):
                    # Get instance IDs for these pixels
                    instance_ids = np.unique(instance_map[vehicle_pixels])
                    # Remove background (0)
                    instance_ids = instance_ids[instance_ids > 0]
                    vehicle_instance_ids.extend(instance_ids)
            
            # Process each vehicle instance
            for i, instance_id in enumerate(vehicle_instance_ids[:LEAD_MHP_N]):
                # Create mask for this instance
                instance_mask = (instance_map == instance_id).astype(np.uint8) * 255
                
                # Get bounding box and centroid
                contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    centroid_x = x + w/2
                    centroid_y = y + h/2
                    
                    # Normalize coordinates
                    norm_x = (centroid_x / instance_map.shape[1] * 2) - 1
                    norm_y = (centroid_y / instance_map.shape[0] * 2) - 1
                    
                    # Store in lead section
                    lead_offset = LEAD_IDX + i * LEAD_MHP_GROUP_SIZE
                    
                    # Store simple lead vehicle representation
                    # x, y, relative distance, width
                    output[lead_offset + 0] = norm_x
                    output[lead_offset + 1] = norm_y
                    output[lead_offset + 2] = 1.0 - norm_y  # Simple distance estimation
                    output[lead_offset + 3] = w / instance_map.shape[1]  # Normalized width
                    
                    # Set lead probability (logit)
                    lead_prob_offset = LEAD_PROB_IDX + i
                    output[lead_prob_offset] = 2.0  # High confidence (logit form)
        
        # Detect stop lines (where vehicles stop at traffic lights and stop signs)
        traffic_control_mask = np.isin(seg_map, TRAFFIC_CONTROL_CLASSES)
        if np.any(traffic_control_mask):
            # Find traffic lights and signs
            traffic_control_mask = traffic_control_mask.astype(np.uint8) * 255
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(traffic_control_mask)
            
            # Sort components by area (excluding background which is label 0)
            sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
            
            # Process up to STOP_LINE_MHP_N traffic control objects
            stop_line_count = 0
            for idx in sorted_indices[:STOP_LINE_MHP_N]:
                # Get component mask
                component_mask = (labels == idx).astype(np.uint8) * 255
                
                # Get bounding box and bottom center
                x = stats[idx, cv2.CC_STAT_LEFT]
                y = stats[idx, cv2.CC_STAT_TOP]
                w = stats[idx, cv2.CC_STAT_WIDTH]
                h = stats[idx, cv2.CC_STAT_HEIGHT]
                
                # Calculate bottom center (where the stop line would be)
                bottom_center_x = x + w/2
                bottom_center_y = y + h  # Bottom of the traffic light/sign
                
                # Check if the traffic control object is on/near a road
                # Define a region below the traffic control object
                stop_line_y = min(bottom_center_y + 20, seg_map.shape[0] - 1)  # 20 pixels below
                
                # Create a horizontal line at this position
                line_start_x = max(0, bottom_center_x - w)
                line_end_x = min(seg_map.shape[1] - 1, bottom_center_x + w)
                
                # Check if this line intersects with road
                line_points_x = np.linspace(line_start_x, line_end_x, 20).astype(int)
                line_points_y = np.ones_like(line_points_x) * stop_line_y
                
                # If we have road pixels along this line, it's a valid stop line
                road_pixels_count = 0
                for px, py in zip(line_points_x, line_points_y):
                    if py < seg_map.shape[0] and px < seg_map.shape[1]:
                        if np.isin(seg_map[py, px], ROAD_CLASSES):
                            road_pixels_count += 1
                
                # If enough road pixels, consider it a valid stop line
                if road_pixels_count > 5:
                    # Normalize coordinates
                    norm_center_x = (bottom_center_x / seg_map.shape[1] * 2) - 1
                    norm_center_y = (stop_line_y / seg_map.shape[0] * 2) - 1
                    norm_width = w / seg_map.shape[1] * 2
                    
                    # Store stop line information
                    stop_line_offset = STOP_LINE_IDX + stop_line_count * STOP_LINE_MHP_GROUP_SIZE
                    
                    # Store stop line parameters (x, y, width, orientation, etc.)
                    output[stop_line_offset + 0] = norm_center_x  # x center
                    output[stop_line_offset + 1] = norm_center_y  # y position (where to stop)
                    output[stop_line_offset + 2] = norm_width  # width
                    output[stop_line_offset + 3] = 0.0  # horizontal line (0 degree rotation)
                    output[stop_line_offset + 4] = 1.0 - norm_center_y  # distance
                    # Other parameters default to 0
                    
                    # Store standard deviations (log values)
                    std_offset = stop_line_offset + STOP_LINE_PRED_DIM
                    output[std_offset + 0] = np.log(0.1)  # x uncertainty
                    output[std_offset + 1] = np.log(0.1)  # y uncertainty
                    output[std_offset + 2] = np.log(0.2)  # width uncertainty
                    
                    # Set probability (logit form)
                    prob_offset = STOP_LINE_PROB_IDX + stop_line_count
                    output[prob_offset] = 2.0  # High confidence
                    
                    stop_line_count += 1
                    if stop_line_count >= STOP_LINE_MHP_N:
                        break
        
        # Default values for other outputs
        
        # Traffic convention (default to left-hand drive)
        desire_input = np.zeros(DESIRE_LEN)
        desire_input[0] = 1.0  # Default "none" desire
        
        # Add these values to the output
        desire_state_offset = DESIRE_STATE_IDX
        for i in range(DESIRE_LEN):
            output[desire_state_offset + i] = desire_input[i]
        
        return output
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get annotation paths
        instance_path, semantic_path = self._get_annotation_path(img_path)
        
        # Load annotations
        instance_map = None
        if instance_path and instance_path.exists():
            instance_map = np.array(Image.open(instance_path))
            
        semantic_map = None
        if semantic_path and semantic_path.exists():
            semantic_map = np.array(Image.open(semantic_path))
        else:
            # If no annotation found, create a dummy one
            semantic_map = np.zeros((image.height, image.width), dtype=np.uint8)
        
        # Resize image to expected input size
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT))
        
        # Also resize the segmentation maps if they exist
        if semantic_map is not None:
            semantic_map = cv2.resize(semantic_map, (MODEL_WIDTH, MODEL_HEIGHT), 
                                       interpolation=cv2.INTER_NEAREST)
        if instance_map is not None:
            instance_map = cv2.resize(instance_map, (MODEL_WIDTH, MODEL_HEIGHT), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Process segmentation into expected output format
        target = self._process_segmentation(semantic_map, instance_map)
        target = torch.from_numpy(target)
        
        # Create default inputs
        desire_input = np.zeros(DESIRE_LEN, dtype=np.float32)
        desire_input[0] = 1.0  # Default to 'none'
            
        traffic_convention = np.zeros(TRAFFIC_CONVENTION_LEN, dtype=np.float32)
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
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing Cityscapes dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples (for debugging)')
    parser.add_argument('--annotation_mode', type=str, default='fine', choices=['fine', 'coarse'], help='Use fine or coarse annotations')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # We'll convert to YUV in the forward pass if needed
    ])
    
    # Create datasets
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
