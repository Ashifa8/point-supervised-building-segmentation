import numpy as np
import cv2
import torch
import rasterio
import geopandas as gpd
from rasterio import features
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG
from sampling import make_point_mask

IGNORE = CFG['ignore_index']


def read_image(path: str) -> np.ndarray:
    """Read a GeoTIFF using rasterio and return as uint8 RGB numpy array."""
    with rasterio.open(path) as src:
        if src.count >= 3:
            img = src.read([1, 2, 3])
        else:
            ch = src.read(1)
            img = np.stack([ch, ch, ch])

    img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    return cv2.resize(img, CFG['img_size'])


def read_mask_from_geojson(geojson_path: str, native_shape: tuple = (1024, 1024)) -> np.ndarray:
    """Rasterise building polygons into a binary mask (0=background, 1=building)."""
    gdf = gpd.read_file(geojson_path)

    if gdf.empty:
        mask = np.zeros(native_shape, dtype=np.uint8)
    else:
        mask = features.rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=native_shape,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

    return cv2.resize(mask, CFG['img_size'], interpolation=cv2.INTER_NEAREST)


def collect_pairs(root: Path):
    """Scan SpaceNet-7 AOI folders and collect matched image-mask pairs."""
    image_paths, mask_paths = [], []
    aois = [p for p in root.iterdir() if p.is_dir()]
    print(f'Scanning {len(aois)} AOI folders ...')

    for aoi in sorted(aois):
        img_dir  = aoi / 'images_masked'
        mask_dir = aoi / 'labels_match_pix'
        if not img_dir.exists() or not mask_dir.exists():
            continue

        mask_dict = {
            m.stem.replace('_Buildings', ''): m
            for m in mask_dir.glob('*.geojson')
        }

        for img_p in sorted(img_dir.glob('*.tif')):
            if img_p.stem in mask_dict:
                image_paths.append(str(img_p))
                mask_paths.append(str(mask_dict[img_p.stem]))

    return image_paths, mask_paths


def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


class SpaceNet7Dataset(Dataset):

    def __init__(self, img_paths, mask_paths, strategy='boundary',
                 ratio=0.01, transform=None, is_val=False):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.strategy   = strategy
        self.ratio      = ratio
        self.transform  = transform
        self.is_val     = is_val

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img       = read_image(self.img_paths[idx])
        full_mask = read_mask_from_geojson(self.mask_paths[idx])

        if self.is_val:
            sup_mask = full_mask.astype(np.int64)
        else:
            sup_mask = make_point_mask(full_mask, self.strategy, self.ratio)

        if self.transform is not None:
            aug      = self.transform(image=img, mask=sup_mask.astype(np.int32))
            img      = aug['image']
            sup_mask = aug['mask'].long()
        else:
            img      = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            sup_mask = torch.from_numpy(sup_mask.astype(np.int64))

        return img, sup_mask


def prepare_data(train_dir: Path):
    all_imgs, all_masks = collect_pairs(train_dir)
    print(f'Total matched pairs: {len(all_imgs)}')

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_imgs, all_masks,
        test_size=CFG['val_split'],
        random_state=CFG['seed']
    )
    print(f'Train: {len(train_imgs)}   Val: {len(val_imgs)}')
    return train_imgs, val_imgs, train_masks, val_masks
