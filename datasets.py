import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import json
from constants import *

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


class Comma10kDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', limit_samples=None):
        """
        Comma10k dataset for driving model training
        
        Args:
            data_dir: Root directory of comma10k dataset
            transform: Optional transforms to apply
            split: 'train' or 'val' 
            limit_samples: Optional limit to dataset size
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # All image files in the repository
        img_paths = []
        # Regular images
        img_paths.extend(list(self.data_dir / 'imgs').glob('*.png'))
        # Fisheye images
        if (self.data_dir / 'imgs2').exists():
            img_paths.extend(list(self.data_dir / 'imgs2').glob('*.png'))
        # Driver camera images
        if (self.data_dir / 'imgsd').exists():
            img_paths.extend(list(self.data_dir / 'imgsd').glob('*.png'))
            
        print(f"Found {len(img_paths)} total images")
        
        # Create train/val split (90/10)
        if split == 'train':
            # Files ending with 9.png are used for validation
            self.image_files = [p for p in img_paths if not p.stem.endswith('9')]
        else:
            # Validation set - files ending with 9.png
            self.image_files = [p for p in img_paths if p.stem.endswith('9')]
        
        print(f"Using {len(self.image_files)} images for {split}")
        
        # Limit dataset size if requested
        if limit_samples:
            self.image_files = self.image_files[:limit_samples]
    
    def __len__(self):
        return len(self.image_files)
    
    def _get_mask_path(self, img_path):
        """Get corresponding mask path for an image"""
        img_name = img_path.name
        
        # Check which images directory this is from
        if 'imgs/' in str(img_path):
            mask_path = self.data_dir / 'masks' / img_name
        elif 'imgs2/' in str(img_path):
            mask_path = self.data_dir / 'masks2' / img_name
        elif 'imgsd/' in str(img_path):
            mask_path = self.data_dir / 'masksd' / img_name
        else:
            return None
            
        if mask_path.exists():
            return mask_path
        
        return None
    
    def _process_comma10k_mask(self, mask):
        """Process comma10k mask into our model's expected output format"""
        # Initialize a full output array with zeros
        output = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        
        # Convert mask from RGB to class indices
        # Comma10k uses RGB colors to represent classes:
        # 1: #402020 (64, 32, 32) - road
        # 2: #ff0000 (255, 0, 0) - lane markings
        # 3: #808060 (128, 128, 96) - undrivable
        # 4: #00ff66 (0, 255, 102) - movable (vehicles and people)
        # 5: #cc00ff (204, 0, 255) - my car
        # 6: #00ccff (0, 204, 255) - movable in car
        
        # Create masks for each class
        road_mask = np.all(mask == np.array([64, 32, 32]), axis=2)
        lane_mask = np.all(mask == np.array([255, 0, 0]), axis=2)
        undrivable_mask = np.all(mask == np.array([128, 128, 96]), axis=2)
        movable_mask = np.all(mask == np.array([0, 255, 102]), axis=2)
        car_mask = np.all(mask == np.array([204, 0, 255]), axis=2)
        
        # Process road edges using the boundary between road and non-road
        if np.any(road_mask):
            road_mask_uint8 = road_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(road_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use largest contour for road edge
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Sample points along contour for road edges
                n_points = min(TRAJECTORY_SIZE, len(approx_contour))
                if n_points > 0:
                    indices = np.linspace(0, len(approx_contour) - 1, n_points).astype(int)
                    
                    # Store road edge points
                    for i, idx in enumerate(indices[:TRAJECTORY_SIZE]):
                        if i < len(indices):
                            point = approx_contour[idx][0]
                            # Normalize coordinates to [-1, 1]
                            x = (point[0] / mask.shape[1] * 2) - 1
                            y = (point[1] / mask.shape[0] * 2) - 1
                            
                            edge_offset = RE_IDX + i * 2
                            output[edge_offset + 0] = x
                            output[edge_offset + 1] = y
        
        # Process lane lines using the lane markings
        if np.any(lane_mask):
            lane_mask_uint8 = lane_mask.astype(np.uint8) * 255
            
            # Skeletonize to get centerlines
            lane_skeleton = cv2.ximgproc.thinning(lane_mask_uint8)
            
            # Split into left/right lane lines (simplified approach: left half/right half)
            h, w = lane_skeleton.shape
            left_half = np.zeros_like(lane_skeleton)
            right_half = np.zeros_like(lane_skeleton)
            
            left_half[:, :w//2] = lane_skeleton[:, :w//2]
            right_half[:, w//2:] = lane_skeleton[:, w//2:]
            
            # Process left lane line
            if np.any(left_half):
                y_indices, x_indices = np.where(left_half > 0)
                if len(y_indices) > 0:
                    # Sort bottom to top
                    sort_idx = np.argsort(y_indices)[::-1]
                    y_sorted = y_indices[sort_idx]
                    x_sorted = x_indices[sort_idx]
                    
                    # Sample points
                    n_points = min(TRAJECTORY_SIZE, len(y_sorted))
                    if n_points > 0:
                        indices = np.linspace(0, len(y_sorted) - 1, n_points).astype(int)
                        
                        # Store left lane line
                        for i, idx in enumerate(indices[:TRAJECTORY_SIZE]):
                            if i < len(indices):
                                # Normalize coordinates
                                x = (x_sorted[idx] / mask.shape[1] * 2) - 1
                                y = (y_sorted[idx] / mask.shape[0] * 2) - 1
                                
                                ll_offset = LL_IDX + i * 2
                                output[ll_offset + 0] = x
                                output[ll_offset + 1] = y
                        
                        # Set high probability for this lane line
                        output[LL_PROB_IDX] = 2.0  # Logit for high probability
            
            # Process right lane line
            if np.any(right_half):
                y_indices, x_indices = np.where(right_half > 0)
                if len(y_indices) > 0:
                    # Sort bottom to top
                    sort_idx = np.argsort(y_indices)[::-1]
                    y_sorted = y_indices[sort_idx]
                    x_sorted = x_indices[sort_idx]
                    
                    # Sample points
                    n_points = min(TRAJECTORY_SIZE, len(y_sorted))
                    if n_points > 0:
                        indices = np.linspace(0, len(y_sorted) - 1, n_points).astype(int)
                        
                        # Store right lane line
                        for i, idx in enumerate(indices[:TRAJECTORY_SIZE]):
                            if i < len(indices):
                                # Normalize coordinates
                                x = (x_sorted[idx] / mask.shape[1] * 2) - 1
                                y = (y_sorted[idx] / mask.shape[0] * 2) - 1
                                
                                ll_offset = LL_IDX + TRAJECTORY_SIZE * 2 + i * 2  # Offset for second lane line
                                output[ll_offset + 0] = x
                                output[ll_offset + 1] = y
                        
                        # Set high probability for this lane line
                        output[LL_PROB_IDX + 2] = 2.0  # Logit for high probability
        
        # Path planning using road center
        if np.any(road_mask):
            road_mask_uint8 = road_mask.astype(np.uint8) * 255
            
            # Find center of the road (assuming mostly vertical roads)
            skeleton = cv2.ximgproc.thinning(road_mask_uint8)
            
            if np.any(skeleton):
                y_indices, x_indices = np.where(skeleton > 0)
                
                if len(y_indices) > 0:
                    # Sort from bottom to top
                    sort_idx = np.argsort(y_indices)[::-1]
                    y_sorted = y_indices[sort_idx]
                    x_sorted = x_indices[sort_idx]
                    
                    # Sample points along path
                    n_points = min(TRAJECTORY_SIZE, len(y_sorted))
                    if n_points > 0:
                        indices = np.linspace(0, len(y_sorted) - 1, n_points).astype(int)
                        
                        # Main path (path 0)
                        path_offset = PLAN_IDX
                        
                        prev_x, prev_y = None, None
                        
                        for i, idx in enumerate(indices):
                            if i < TRAJECTORY_SIZE:
                                # Normalize coordinates
                                x = (x_sorted[idx] / mask.shape[1] * 2) - 1
                                y = (y_sorted[idx] / mask.shape[0] * 2) - 1
                                
                                # Store position
                                point_offset = path_offset + i * PLAN_MHP_COLUMNS
                                output[point_offset + 0] = x
                                output[point_offset + 1] = y
                                output[point_offset + 2] = 0  # z=0 for 2D path
                                
                                # Estimate velocity (if not the first point)
                                if i > 0 and prev_x is not None:
                                    dx = x - prev_x
                                    dy = y - prev_y
                                    output[point_offset + 3] = dx * 10  # Scale for velocity
                                    output[point_offset + 4] = dy * 10
                                
                                prev_x, prev_y = x, y
                        
                        # Set high probability for this path
                        path_prob_idx = path_offset + TRAJECTORY_SIZE * PLAN_MHP_COLUMNS * 2
                        output[path_prob_idx] = 2.0  # High confidence logit
        
        # Lead vehicle detection using the movable mask
        if np.any(movable_mask):
            movable_mask_uint8 = movable_mask.astype(np.uint8) * 255
            
            # Find connected components (potential vehicles)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(movable_mask_uint8)
            
            # Sort by area (largest first, excluding background)
            sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
            
            # Process up to LEAD_MHP_N lead vehicles
            lead_count = 0
            for idx in sorted_indices[:LEAD_MHP_N]:
                # Get component properties
                x = stats[idx, cv2.CC_STAT_LEFT]
                y = stats[idx, cv2.CC_STAT_TOP]
                w = stats[idx, cv2.CC_STAT_WIDTH]
                h = stats[idx, cv2.CC_STAT_HEIGHT]
                
                # Simple filter for reasonable vehicle size
                if w > 5 and h > 5:
                    # Calculate center
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Normalize coordinates
                    norm_x = (center_x / mask.shape[1] * 2) - 1
                    norm_y = (center_y / mask.shape[0] * 2) - 1
                    
                    # Store lead vehicle parameters
                    lead_offset = LEAD_IDX + lead_count * LEAD_MHP_GROUP_SIZE
                    
                    # Basic lead vehicle representation (x, y, distance, width)
                    output[lead_offset + 0] = norm_x
                    output[lead_offset + 1] = norm_y
                    output[lead_offset + 2] = 1.0 - norm_y  # Simple distance estimation
                    output[lead_offset + 3] = w / mask.shape[1] * 2  # Normalized width
                    
                    # Set high probability for lead detection
                    lead_prob_offset = LEAD_PROB_IDX + lead_count
                    output[lead_prob_offset] = 2.0  # High confidence
                    
                    lead_count += 1
                    if lead_count >= LEAD_MHP_N:
                        break
        
        # For stop lines - comma10k doesn't have explicit stop line annotations,
        # so we'll use default/static values for training
        
        # Set default values for stop lines with low confidence
        for i in range(STOP_LINE_MHP_N):
            stop_line_offset = STOP_LINE_IDX + i * STOP_LINE_MHP_GROUP_SIZE
            
            # Set default values - far ahead in the distance with low confidence
            output[stop_line_offset + 0] = 0.0  # x center (middle)
            output[stop_line_offset + 1] = -0.9  # y position (far ahead)
            output[stop_line_offset + 2] = 0.5  # width
            
            # Set low probability
            prob_offset = STOP_LINE_PROB_IDX + i
            output[prob_offset] = -3.0  # Very low confidence (logit form)
        
        # Default desire and traffic convention values
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
        
        # Get mask path
        mask_path = self._get_mask_path(img_path)
        
        # Load mask
        if mask_path and mask_path.exists():
            mask = np.array(Image.open(mask_path))
        else:
            # Create a dummy mask if none exists
            mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            print(f"Warning: No mask found for {img_path}")
        
        # Resize image to expected input size
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT))
        
        # Resize mask to expected size
        mask = cv2.resize(mask, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Process mask into expected output format
        target = self._process_comma10k_mask(mask)
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