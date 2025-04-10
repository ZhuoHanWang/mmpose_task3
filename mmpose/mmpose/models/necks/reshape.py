from mmpose.registry import MODELS
from mmengine.model import BaseModule
import torch
from typing import Tuple


@MODELS.register_module()
class ReshapeNeck(BaseModule):
    """Reshape Neck to adjust tensor dimensions.

    This module is designed to reshape the tensor dimensions between
    the backbone and the head.

    Args:
        input_shape (Tuple[int]): Expected input shape of the tensor.
        output_shape (Tuple[int]): Desired output shape of the tensor.
    """

    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor to the desired shape."""
        assert x.numel() == torch.prod(torch.tensor(self.output_shape)), (
            f"Reshape error: input elements {x.numel()} do not match "
            f"output elements {torch.prod(torch.tensor(self.output_shape))}")
        return x.view(*self.output_shape)
