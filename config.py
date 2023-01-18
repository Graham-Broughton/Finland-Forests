import os
from dataclasses import dataclass

@dataclass
class CFG:
    BUCKET: str = 'gs://finlandforest'
    TRAIN_SAT: str = os.path.join(BUCKET, 'train_features')
    TRAIN_AGBM: str = os.path.join(BUCKET, 'train_agbm')
    TEST: str = os.path.join(BUCKET, 'test')
    SEED: int = 42
    PATH = 'gcp_bucket/train_features/'

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

    def __post_init__(self):
        tmp = os.path.realpath(__file__)
        not_base_path = True
        while not_base_path:
            if os.path.split(tmp)[1] == 'finland_forest':
                not_base_path = False
                self.BASE = tmp
            else:
                tmp = os.path.split(tmp)[0]
        self.DATA = os.path.join(self.BASE, 'data')
        self.SRC = os.path.join(self.BASE, 'src')
        self.TILE_CSV = os.path.join(self.DATA, 'df_tile_list.csv')

