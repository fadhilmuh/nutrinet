from transformers import DPTConfig, DPTForDepthEstimation, DPTFeatureExtractor
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class DepthBlock(nn.Module):
    def __init__(self, device, state_dict=None, return_rgb=True, single_return=True, training=False):
        super(DepthBlock, self).__init__()
        self.return_rgb = return_rgb
        self.single_return = single_return
        self.device = device

        # Initialize DPT model and feature extractor without downloading weights
        config = DPTConfig.from_pretrained("Intel/dpt-large")
        self.dpt = DPTForDepthEstimation(config).to(device)
        if state_dict is not None:
            self.dpt.load_state_dict(torch.load(state_dict, map_location=device, weights_only=True))
        
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

        # Set trainability
        if training:
            for param in self.dpt.parameters():
                param.requires_grad = True
        else:
            for param in self.dpt.parameters():
                param.requires_grad = False

    def forward(self, images):
        images = images.to(self.device)
        inputs = self.feature_extractor(images=images * 255, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.dpt(**inputs)
        depth_maps = outputs.predicted_depth

        depth_rgb_batch = None
        if self.return_rgb:
            depth_rgb_batch = []
            colormap = plt.get_cmap("jet")

            for depth_map in depth_maps:
                depth_map = 255 * (depth_map - depth_map.min() + 1e-8) / (depth_map.max() - depth_map.min() + 1e-8) # Normalize depth map for rgb conversion
                depth_map_rgb = colormap(depth_map.detach().cpu().numpy().astype(np.uint8))[:, :, [0,1,2]]
                depth_map_rgb = np.transpose(depth_map_rgb, (2, 0, 1))

                depth_map_image = (255 * torch.tensor(depth_map_rgb, dtype=torch.float32)).to(self.device)
                depth_rgb_batch.append(depth_map_image)

            depth_rgb_batch = torch.stack(depth_rgb_batch, dim=0)

        depth_raw_batch = depth_maps.unsqueeze(1)

        if self.single_return and self.return_rgb and depth_rgb_batch is not None:
            return depth_rgb_batch
        elif self.single_return and not self.return_rgb:
            return depth_raw_batch
        elif not self.single_return:
            return depth_rgb_batch, depth_raw_batch
        else:
            raise Exception("Invalid Parameters for Depth Model")