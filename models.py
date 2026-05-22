from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _expand_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight.zero_()
        copy_channels = min(conv.in_channels, in_channels)
        new_conv.weight[:, :copy_channels] = conv.weight[:, :copy_channels]
        if in_channels > conv.in_channels:
            mean_weight = conv.weight.mean(dim=1, keepdim=True)
            for c in range(conv.in_channels, in_channels):
                new_conv.weight[:, c : c + 1] = mean_weight
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


def build_model(name: str, num_classes: int = 3, in_channels: int = 6, pretrained: bool = False):
    weights = None
    if name == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.conv1 = _expand_conv_in_channels(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        first_conv = model.features[0][0]
        model.features[0][0] = _expand_conv_in_channels(first_conv, in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"unsupported model: {name}")
