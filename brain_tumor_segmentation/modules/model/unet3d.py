from monai.networks.nets import UNet


def get_unet3d_model(
    in_channels: int = 1, out_channels: int = 1, dropout: float = 0.1
) -> UNet:
    unet = UNet(
        dimensions=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=dropout,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )

    return unet
