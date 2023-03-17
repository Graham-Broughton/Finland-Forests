from dataclasses import dataclass, field
import os


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
TILES = os.path.join(DATA_PATH, 'train', 'train_features')
GT = os.path.join(DATA_PATH, 'train', 'train_agbm')
TEST = os.path.join(DATA_PATH, 'test')
METADATA = os.path.join(DATA_PATH, 'features_metadata.csv')


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
    BASE_PATH: str = BASE_PATH
    DATA_PATH: str = DATA_PATH
    TILES: str = TILES
    GT: str = GT
    TEST: str = TEST
    METADATA: str = METADATA
