import torch
import torch.nn as nn
from torchvision import models

from model.dpt import DepthBlock

class CrossModalAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossModalAttentionBlock, self).__init__()

        # Channel Attention (CA)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention (SA)
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.spatial_sigmoid = nn.Sigmoid()

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, rgb_feat, depth_feat):
        # Element-wise addition
        combined_feat = rgb_feat + depth_feat

        # Channel Attention (CA)
        ca = self.global_avg_pool(combined_feat)
        ca = self.channel_conv(ca)
        ca = self.channel_sigmoid(ca)
        ca_rgb = rgb_feat * ca
        ca_depth = depth_feat * ca

        # Spatial Attention (SA)
        sa = torch.mean(combined_feat, dim=1, keepdim=True)
        sa = self.spatial_conv(sa)
        sa = self.spatial_sigmoid(sa)
        sa_rgb = ca_rgb * sa
        sa_depth = ca_depth * sa

        fused_feat = torch.cat([sa_rgb, sa_depth], dim=1)
        output = self.final_conv(fused_feat)

        return output

class DPFNutritionNet(nn.Module):
    def __init__(self):
        super(DPFNutritionNet, self).__init__()
        
        resnet_rgb = models.resnet101(pretrained=True)
        self.rgb_layer1 = nn.Sequential(*list(resnet_rgb.children())[:5])  # Up to layer1
        self.rgb_layer2 = nn.Sequential(*list(resnet_rgb.children())[5])   # layer2
        self.rgb_layer3 = nn.Sequential(*list(resnet_rgb.children())[6])   # layer3
        self.rgb_layer4 = nn.Sequential(*list(resnet_rgb.children())[7])   # layer4

        resnet_depth = models.resnet101(pretrained=True)
        
        resnet_depth.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=resnet_depth.conv1.out_channels,
            kernel_size=resnet_depth.conv1.kernel_size,
            stride=resnet_depth.conv1.stride,
            padding=resnet_depth.conv1.padding,
            bias=resnet_depth.conv1.bias is not None
        )
        
        with torch.no_grad():
            resnet_depth.conv1.weight = nn.Parameter(
                resnet_depth.conv1.weight.sum(dim=1, keepdim=True)
            )
        
        self.depth_layer1 = nn.Sequential(*list(resnet_depth.children())[:5])  # Up to layer1
        self.depth_layer2 = nn.Sequential(*list(resnet_depth.children())[5])   # layer2
        self.depth_layer3 = nn.Sequential(*list(resnet_depth.children())[6])   # layer3
        self.depth_layer4 = nn.Sequential(*list(resnet_depth.children())[7])   # layer4

        self.cab1 = CrossModalAttentionBlock(256)
        self.cab2 = CrossModalAttentionBlock(512)
        self.cab3 = CrossModalAttentionBlock(1024)
        self.cab4 = CrossModalAttentionBlock(2048)

        # Fully-connected layer for the final output
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 5)
        )

    def forward(self, rgb, depth):
        # Layer 1 processing
        rgb_feat1 = self.rgb_layer1(rgb)
        depth_feat1 = self.depth_layer1(depth)
        fused_feat1 = self.cab1(rgb_feat1, depth_feat1)

        # Layer 2 processing
        rgb_feat2 = self.rgb_layer2(fused_feat1 + rgb_feat1)
        depth_feat2 = self.depth_layer2(fused_feat1 + depth_feat1)
        fused_feat2 = self.cab2(rgb_feat2, depth_feat2)

        # Layer 3 processing
        rgb_feat3 = self.rgb_layer3(fused_feat2 + rgb_feat2)
        depth_feat3 = self.depth_layer3(fused_feat2 + depth_feat2)
        fused_feat3 = self.cab3(rgb_feat3, depth_feat3)

        # Layer 4 processing
        rgb_feat4 = self.rgb_layer4(fused_feat3 + rgb_feat3)
        depth_feat4 = self.depth_layer4(fused_feat3 + depth_feat3)
        fused_feat4 = self.cab4(rgb_feat4, depth_feat4)

        # Final classification
        output = self.fc(fused_feat4)
        return output

class DPFNutritionModel(nn.Module):
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), state_dict_file=None, no_depth=False):
        super(DPFNutritionModel, self).__init__()
        self.dpfnet = DPFNutritionNet().to(device)
        self.depth_model = DepthBlock(device=device, state_dict=state_dict_file, return_rgb=False, single_return=True, training=False)
        self.no_depth = no_depth
        self.device = device

    def forward(self, rgb_images, depths=None):
        rgb_images = rgb_images.to(self.device)
        if self.no_depth:
            depths = self.depth_model(rgb_images) / 255.
            depths = depths.to(self.device)
        else:
            if depths is not None:
                depths = depths.to(self.device)
            else:
                raise ValueError("Depth images must be provided when no_depth is False.")
        
        return self.dpfnet(rgb_images, depths)