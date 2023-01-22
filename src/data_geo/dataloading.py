import pandas as pd
import rasterio
import src.data_geo.transforms as tf
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from glob import glob
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SentinelDataset(Dataset):
    def __init__(
            self, 
            tile_file, 
            dir_tiles, 
            dir_target, 
            metadata,
            path,
            max_chips=None, 
            transform=None, 
            device='cpu', 
            is_train=True
        ):
        self.metadata = pd.read_csv(metadata)
        self.path = path
        self.dir_tiles = dir_tiles
        self.dir_target = dir_target
        self.device = device        
        if tile_file is not None:
            self.df_tile_list = pd.read_csv(tile_file)[['chipid', 'month']]
        else:
            self.df_tile_list = self._make_tile_file()
        if max_chips:
            self.df_tile_list = self.df_tile_list[:max_chips]
        # self.df_tile_list = self.df_tile_list[['chip_id', 'month']].to_records(index=False).tolist()
        self.transform_s2 = tf.Sentinel2Scale()
        self.transform_s1 = tf.Sentinel1Scale()
        self.transform = transform
        

    def _make_tile_file(self):
        if self.dir_target:
            files = self.metadata[self.metadata['split'] == 'train']
            filename = 'train_tiles.csv'
        else:
            files = self.metadata[self.metadata['split'] == 'test']
            filename = 'test_tiles.csv'

        files = pd.DataFrame(
                    files['filename'].apply(
                        lambda x: str(x).split('.')[0].split("_")[::2]
                    ).tolist(), columns=['chip_id', 'month']
                ).drop_duplicates()
        files['month'] = files['month'].astype(int)
        files = files.sort_values(['chip_id', 'month']).reset_index(drop=True)
        files.to_csv(f'{self.path}/{filename}', index=False)
        return files

    def _load_agbm_tile(self, chipid):
        target_path = os.path.join(self.dir_target, f'{chipid}_agbm.tif')
        return self._read_tif_to_tensor(target_path)

    def _read_tif_to_tensor(self, tif_path):
        with rasterio.open(tif_path) as src:
            X = torch.tensor(src.read().astype(np.float32),
                             dtype=torch.float32,
                             device=self.device,
                             requires_grad=False,
                             )
        return X

    def _load_sentinel_tiles(self, sentinel_type, chipid, month):
        file_name = f'{chipid}_{sentinel_type}_{str(month).zfill(2)}.tif'
        tile_path = os.path.join(self.dir_tiles, file_name)
        return self._read_tif_to_tensor(tile_path)

    def __len__(self):
        return len(self.df_tile_list)

    def __getitem__(self, index):
        chipid, month = self.df_tile_list.iloc[index].values

        try:
            s1_tile = self._load_sentinel_tiles('S1', chipid, month)
            s1_tile_scaled = self.transform_s1(s1_tile)
        except Exception as e:
            # print(f'Data load failure for S1: {chipid} {month}')
            # print(f'Caused by {e}')
            s1_tile_scaled = torch.full([4, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)
        # Sentinel 2
        try:
            s2_tile = self._load_sentinel_tiles('S2', chipid, month)
            s2_tile_scaled = self.transform_s2(s2_tile)
        except Exception as e:
            # print(f'Data load failure for S2: {chipid} {month}')
            # print(f'Caused by {e}')
            s2_tile_scaled = torch.full([11, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)

        tile = torch.cat([s1_tile_scaled, s2_tile_scaled], dim=0)

        if self.dir_target is not None:
            target_tile = self._load_agbm_tile(chipid)
        else:
            target_tile = torch.full([1, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)

        sample = {'image': tile, 'label': target_tile} # 'image' and 'label' are used by torchgeo

        if self.transform:
            sample = self.transform(sample)

        return sample