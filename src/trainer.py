import gc
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import SpaceNet7Dataset, get_transforms
from model import build_model, DEVICE
from loss import PartialCrossEntropyLoss, compute_iou

IGNORE = CFG['ignore_index']


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch:02d} [Train]', leave=True,
                bar_format='{l_bar}{bar:30}{r_bar}')

    for step, (images, masks) in enumerate(pbar, 1):
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f'{running_loss / step:.4f}')

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    pbar = tqdm(loader, desc='           [Val]  ', leave=True,
                bar_format='{l_bar}{bar:30}{r_bar}')

    for images, masks in pbar:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        logits     = model(images)
        loss       = F.cross_entropy(logits, masks, ignore_index=IGNORE)
        iou        = compute_iou(logits, masks, CFG['num_classes'], IGNORE)

        total_loss += loss.item()
        total_iou  += iou

        pbar.set_postfix(loss=f'{loss.item():.4f}', mIoU=f'{iou:.4f}')

    n = len(loader)
    return total_loss / n, total_iou / n


def run_experiment(run_name, train_imgs, val_imgs,
                   train_masks, val_masks,
                   strategy='boundary', ratio=0.01, epochs=None):
    """Train one experiment configuration and return history + best mIoU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    epochs = epochs or CFG['epochs']
    print(f'\nStarting run: {run_name}')
    print(f'Strategy: {strategy}  |  Ratio: {ratio * 100:.1f}%')
    print('-' * 45)

    train_transform, val_transform = get_transforms()

    train_ds = SpaceNet7Dataset(train_imgs, train_masks, strategy=strategy,
                                ratio=ratio, transform=train_transform, is_val=False)
    val_ds   = SpaceNet7Dataset(val_imgs,   val_masks,   strategy=strategy,
                                ratio=ratio, transform=val_transform,   is_val=True)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                              num_workers=CFG['num_workers'], pin_memory=True,
                              persistent_workers=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False,
                              num_workers=CFG['num_workers'], pin_memory=True,
                              persistent_workers=False)

    model     = build_model()
    criterion = PartialCrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'],
                                 weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=epochs, eta_min=1e-6)

    history  = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        tr_loss  = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_iou = validate(model, val_loader)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CFG['save_dir'] / f'{run_name}_best.pth')

        print(f'Epoch {epoch:02d}/{epochs} | train_loss {tr_loss:.4f} | '
              f'val_loss {val_loss:.4f} | val_mIoU {val_iou:.4f} | {time.time()-t0:.0f}s')

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'Best val mIoU for {run_name}: {best_iou:.4f}')
    return history, best_iou
