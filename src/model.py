import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import CFG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()


def build_model(encoder: str = 'efficientnet-b3', pretrained: bool = True) -> nn.Module:
    """
    Build DeepLabV3+ with EfficientNet-B3 encoder.
    Automatically wraps in DataParallel when multiple GPUs are available.
    """
    model = smp.DeepLabV3Plus(
        encoder_name    = encoder,
        encoder_weights = 'imagenet' if pretrained else None,
        in_channels     = 3,
        classes         = CFG['num_classes'],
    )

    if N_GPUS > 1:
        print(f'Using DataParallel on {N_GPUS} GPUs')
        model = nn.DataParallel(model)

    return model.to(DEVICE)
