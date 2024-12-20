import torch
from torch.autograd import Variable
from torch.nn.functional import cross_entropy, softmax


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)

    from pytorch3dunet:
    https://github.com/wolny/pytorch-3dunet/

    See original license file:
    https://github.com/wolny/pytorch-3dunet/blob/master/LICENSE
    """
    # number of channels
    n_channels = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(n_channels, -1)


class WeightedCrossEntropyLoss(torch.nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in:
    https://arxiv.org/pdf/1707.03237.pdf

    from pytorch3dunet:
    https://github.com/wolny/pytorch-3dunet/

    See original license file:
    https://github.com/wolny/pytorch-3dunet/blob/master/LICENSE
    """

    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input:  torch.Size([2, 3, 32, 16, 28]) , target (init):  torch.Size([2, 1, 32, 16, 28])
        input:  torch.Size([2, 3, 32, 16, 28]) , target (squeezed):  torch.Size([2, 32, 16, 28])
        """
        weight = self._class_weights(input)
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()
        return cross_entropy(
            input, target, weight=weight, ignore_index=self.ignore_index
        )

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights