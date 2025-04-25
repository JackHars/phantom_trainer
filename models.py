import torch
import torch.nn as nn

# Constants
TEMPORAL_SIZE = 512  # Temporal feature size

# Define a proper ResNet BasicBlock
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class SuperComboNet(nn.Module):
    def __init__(self, temporal_size=TEMPORAL_SIZE, output_size=None):
        super(SuperComboNet, self).__init__()
        
        self.output_size = output_size
        
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
            nn.Linear(1024, output_size)
        )
        
        # Initialize temporal feature
        self.temporal_feature = torch.zeros(1, 1, temporal_size)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block handles downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # First block with potential downsample
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks (no stride/downsample)
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def reset_temporal_state(self, batch_size=1):
        self.temporal_feature = torch.zeros(batch_size, 1, self.gru.hidden_size, device=next(self.parameters()).device)
    
    def forward(self, image, desire=None, traffic_convention=None, recurrent_state=None):
        batch_size = image.size(0)
        
        # CNN feature extraction
        x = self.encoder(image)
        x = torch.mean(x, dim=(2, 3))  # Global average pooling
        
        # Process temporal features - handle as input for ONNX compatibility
        if recurrent_state is not None:
            current_state = recurrent_state
        else:
            # Always create a new zero state if none provided
            # This is more ONNX-friendly than checking tensor size
            current_state = torch.zeros(batch_size, 1, self.gru.hidden_size, 
                                      device=next(self.parameters()).device)
        
        # Process recurrent component
        x_gru = x.unsqueeze(1)  # Add sequence dimension
        gru_out, h_n = self.gru(x_gru, current_state.transpose(0, 1).contiguous())
        
        # Store new state for next iteration
        new_state = h_n.transpose(0, 1)
        
        # Feature fusion
        combined = torch.cat([x, gru_out.squeeze(1)], dim=1)
        fused = self.fusion(combined)
        
        # Generate flat output array
        output = self.output_head(fused)
        
        # Return the primary output and current state
        return output, new_state

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