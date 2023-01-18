import pandas as pd
import numpy as np
import rasterio
import io
from glob import glob
from drive.MyDrive.finland_forest.config import CFG
CFG = CFG()

class Sentinel:
    def __init__(
                self, tile_csv=CFG.TILE_CSV, tif_dir=CFG.TRAIN_SAT,
                dir_target=CFG.TRAIN_AGBM, max_chips=None, transform=None,
                shuffle=True
            ):
        self.tif_dir = tif_dir
        self.dir_target = dir_target
        self.transform  = transform
        self.shuffle = shuffle        
        if tile_csv:
            self.df_tile_list = pd.read_csv(tile_csv).sample(
                frac=1, random_state=CFG.SEED
                ).reset_index(drop=True)
        else:
            self.df_tile_list = self.make_tiles(tif_dir).sample(
                frac=1, random_state=CFG.SEED
                ).reset_index(drop=True)
        if max_chips:
            self.df_tile_list = self.df_tile_list[:max_chips].sample(
                frac=1, random_state=CFG.SEED
                ).reset_index(drop=True)

    
    def __len__(self):
        return len(self.df_tile_list)
    
    def __getitem__(self, idx):
        chipid, month = self.df_tile_list.iloc[idx].values
        # Sent 1
        try:
            s1_tile = self.load_tiles('S1', chipid, month)
            # s1_tile_scaled = self.transform_s1(s1_tile)
        except Exception as e:
            print(f"Error: {e}")
            s1_tile = np.empty((4, 256, 256))
            s1_tile[:] = np.nan
        try:
            s2_tile = self.load_tiles('S2', chipid, month)
            # s2_tile_scaled = self.transform_s2(s2_tile)
        except Exception as e:
            print(f"Error: {e}")
            s2_tile = np.empty((11, 256, 256))
            s2_tile[:] = np.nan
        
        sentinel_tile = np.concatenate([s1_tile, s2_tile], axis=0)

        if self.dir_target:
            target_tile = self.load_agbm_tiles(chipid)
        else:
            target_tile = np.empty((1, 256, 256))
            target_tile[:] = np.nan
        
        sample = {'image': sentinel_tile, 'label': target_tile}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df_tile_list = self.df_tile_list.sample(
                frac=1, random_state=CFG.SEED
                ).reset_index(drop=True)

    def make_tiles(self, tif_dir):
        files = [
            os.path.basename(f).split('.')[0] for f in glob(f'{tif_dir}/*.tif')
        ]
        tile_tuples = []
        for f in files:
            chipid, _, month = f.split('_')
            tile_tuples.append(tuple([chipid, int(month)]))
        tile_tuples = list(set(tile_tuples))
        tile_tuples.sort()
        return pd.DataFrame(tile_tuples, columns=['chipid', 'month'])

    def read_tif(self, tif_path):
        with rasterio.open(tif_path) as f:
            return f.read().astype(np.float32)

    def load_tiles(self, sentinel, chipid, month):
        file_name = f'{chipid}_{sentinel}_{str(month).zfill(2)}.tif'
        file_path = os.path.join(self.tif_dir, file_name)
        return self.read_tif(file_path)
    
    def load_agbm_tiles(self, chipid):
        target_path = os.path.join(self.dir_target, f'{chipid}_agbm.tif')
        return self.read_tif(target_path)