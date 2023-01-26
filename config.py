from dataclasses import dataclass, field


@dataclass
class CFG:
    encoder_widths: list = field(default_factory=[64, 64, 64, 128])
    decoder_widths: list = field(default_factory=[32, 32, 64, 128])
    out_conv: list = field(default_factory=[32, 20])
    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1
    agg_mode: str = "att_group"
    encoder_norm: str = "group"
    n_head: int = 16
    d_model: int = 256
    d_k: int = 4
    num_workers: int = 8
    display_step: int = 100
    epochs: int = 1
    batch_size: int = 32
    lr: float = 0.001
    fold: int = 1
    num_classes: int = 1
    ignore_index: int = -1
    pad_value: int = 0
    padding_mode: str = 'reflect'
    val_every: int = 1
    val_after: int = 0
