import os
from dataclasses import dataclass, asdict, field


@dataclass
class CFG:
    SEED: int = 42
    BATCH_SIZE: int = 64
    WORKERS: int = 14
    EPOCHS: int = 50
    LR: int = 0.02
    PREFETCH: int = 2
    BANDS: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21])
    TRAIN_FRAC: float = 0.8
    MAX_CHIPS: int = None
    WORKING_DIR: str = '/home/broug/Desktop/DD-Finland-Forests'
    DIR_TILES: str = f'{WORKING_DIR}/data/train/train_features/'
    DIR_TARGET: str = f'{WORKING_DIR}/data/train/train_agbm/'
    DIR_TEST: str = f'{WORKING_DIR}/data/test/test_features/'
    SAVED_MODELS: str = f'{WORKING_DIR}/trained_models/'
    TILE_FILE: str = f'{WORKING_DIR}/data/TILE_LIST_BEST_MONTHS.csv'
    TILE_FILE_TEST: str = f'{WORKING_DIR}/data/TILE_LIST_BEST_TEST_MONTHS.csv'
    DIR_PREDS: str = f'{WORKING_DIR}/predictions/baseline/'

    def todict(self):
        return asdict(self)


@dataclass
class CFG2:

    BAND_MAP = {  
        # S2 bands
        0: 'S2-B2: Blue-10m',
        1: 'S2-B3: Green-10m',
        2: 'S2-B4: Red-10m',
        3: 'S2-B5: VegRed-704nm-20m',
        4: 'S2-B6: VegRed-740nm-20m',
        5: 'S2-B7: VegRed-780nm-20m',
        6: 'S2-B8: NIR-833nm-10m',
        7: 'S2-B8A: NarrowNIR-864nm-20m',
        8: 'S2-B11: SWIR-1610nm-20m',
        9: 'S2-B12: SWIR-2200nm-20m',
        10: 'S2-CLP: CloudProb-160m',
        # S1 bands
        11: 'S1-VV-Asc: Cband-10m',
        12: 'S1-VH-Asc: Cband-10m',
        13: 'S1-VV-Desc: Cband-10m',
        14: 'S1-VH-Desc: Cband-10m',
        # Bands derived by transforms 
        15: 'S2-NDVI: (NIR-Red)/(NIR+Red) 10m',
        16: 'S1-NDVVVH-Asc: Norm Diff VV & VH, 10m',
        17: 'S2-NDBI: Difference Built-up Index, 20m',
        18: 'S2-NDRE: Red Edge Vegetation Index, 20m',
        19: 'S2-NDSI: Snow Index, 20m',
        20: 'S2-NDWI: Water Index, 10m',
        21: 'S2-SWI: Sandardized Water-Level Index, 20m',
        22: 'S1-VV/VH-Asc: Cband-10m',
        23: 'S2-VV/VH-Desc: Cband-10m'
}

    MONTH_MAP = {
        0: 'Sep', 1: 'Oct', 2: 'Nov', 3: 'Dec',
        4: 'Jan', 5: 'Feb', 6: 'Mar', 7: 'Apr',
        8: 'May', 9: 'Jun', 10: 'Jul', 11: 'Aug'
}

    CHANNEL_MAP = {
    0: 'S2-B2: Blue-10m',
    1: 'S2-B3: Green-10m',
    2: 'S2-B4: Red-10m',
    3: 'S2-B5: VegRed-704nm-20m',
    4: 'S2-B6: VegRed-740nm-20m',
    5: 'S2-B7: VegRed-780nm-20m',
    6: 'S2-B8: NIR-833nm-10m',
    7: 'S2-B8A: NarrowNIR-864nm-20m',
    8: 'S2-B11: SWIR-1610nm-20m',
    9: 'S2-B12: SWIR-2200nm-20m',
    10: 'S2-CLP: CloudProb-160m',
    11: 'S1-VV-Asc: Cband-10m',
    12: 'S1-VH-Asc: Cband-10m',
    13: 'S1-VV-Desc: Cband-10m',
    14: 'S1-VH-Desc: Cband-10m',
    15: 'S2-NDVI: (NIR-Red)/(NIR+Red) 10m',
    16: 'S1-VV/VH-Asc: Cband-10m'
}

    PLOT_CHANNELS = {
    0: {'s2_rgb_idxs': [2, 1, 0], 'title': 'RGB'}, # RGB image, special case 
    1: {'data': 'target', 'title': 'AGBM', 'LogNorm': True}, # AGBM target data, special case 
    2: {'channel_idx': 6},  # general case of visualizing by channel_index 
    3: {'channel_idx': 15},
    4: {'channel_idx': 10},
    5: {'channel_idx': 12}, 
    6: {'channel_idx': 16},
    7: {'s1_rgb_idxs': [11, 12, 16], 'title': 'SAR-RGB: VV,VH,VV/VH'}, # SAR psuedo-RGB, special case 
}
