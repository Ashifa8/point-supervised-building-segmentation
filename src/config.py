from pathlib import Path

CFG = {
    'ignore_index'     : 255,
    'num_classes'      : 2,
    'img_size'         : (512, 512),
    'epochs'           : 5,
    'batch_size'       : 8,
    'lr'               : 1e-4,
    'weight_decay'     : 1e-5,
    'val_split'        : 0.15,
    'num_workers'      : 2,
    'default_ratio'    : 0.01,
    'default_strategy' : 'boundary',
    'save_dir'         : Path('./outputs'),
    'seed'             : 42,
}

CFG['save_dir'].mkdir(parents=True, exist_ok=True)
