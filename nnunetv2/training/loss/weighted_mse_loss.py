from torch import nn
import torch.nn.functional as F

class WeightedMSELoss(nn.MSELoss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        """
        input:  torch.Size([2, 3, 256, 128, 224]) , target:  torch.Size([2, 1, 256, 128, 224])
        
        """
        # Pop channel dimension if present
        if target.ndim >= input.ndim:
            # Convert target to [Batch, Depth, Height, Width] from [Batch, Channel, Depth, Height, Width]
            assert target.shape[1] == 1
            target = target[:, 0] 
        target = target.long() # Shape: [Batch, Depth, Height, Width]
        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=3)  # Shape: [Batch, Depth, Height, Width, num_classes]
        # Permute dimensions to match input shape: [Batch, Channels, Depth, Height, Width]
        target = target.permute(0, 4, 1, 2, 3).float()  # Shape: [Batch, num_classes, Depth, Height, Width]
        # # Apply sigmoid activation to input to ensure values are between 0 and 1
        input = F.sigmoid(input)
        # # Apply thresholding to input
        # input = torch.where(input < 0.5, torch.zeros_like(input), input)
        
        # Calculate MSE loss, requires input and target to have the same shape: [Batch, num_classes, Depth, Height, Width], the input is expected to be the output of a sigmoid function
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()
