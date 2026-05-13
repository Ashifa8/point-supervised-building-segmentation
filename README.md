# Point-Supervised Building Segmentation

Semantic segmentation of satellite imagery using **only sparse point labels** instead of full pixel masks — a weak supervision approach that drastically reduces annotation cost.

Built with **DeepLabV3+** (EfficientNet-B3 encoder) and trained using **Partial Cross-Entropy Loss** on the [SpaceNet-7](https://spacenet.ai/sn7-challenge/) dataset.

---

## Results

| Experiment | Configuration | Best Val mIoU |
|---|---|---|
| Sampling Ratio | 0.5% (~1,300 pts) | 0.4632 |
| Sampling Ratio | 1.0% (~2,600 pts) | 0.4676 |
| Sampling Ratio | 2.0% (~5,200 pts) | 0.4582 |
| Sampling Ratio | 5.0% (~13,100 pts) | 0.4693 |
| Sampling Strategy | Random (1%) | **0.5679** |
| Sampling Strategy | Boundary-Aware (1%) | 0.4759 |

**Key Finding:** Sampling *strategy* matters far more than sampling *ratio*. Random sampling outperformed boundary-aware at low annotation budgets (0.092 mIoU difference), while changing the ratio from 0.5% to 5% only produced a 0.006 difference.

---

## Project Structure

```
point-supervised-building-seg/
├── src/
│   ├── config.py        # All hyperparameters and paths
│   ├── dataset.py       # SpaceNet-7 data loading & augmentation
│   ├── sampling.py      # Random and boundary-aware point sampling
│   ├── model.py         # DeepLabV3+ with EfficientNet-B3
│   ├── loss.py          # Partial Cross-Entropy Loss + mIoU metric
│   ├── trainer.py       # Train/validation loops & experiment runner
│   ├── visualize.py     # Plotting and CSV export
│   └── main.py          # Entry point
├── notebooks/
│   └── ml-segmentation.ipynb   # Original Kaggle notebook
├── outputs/             # Saved model weights & plots (git-ignored)
├── data/                # Dataset directory (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
git clone https://github.com/your-username/point-supervised-building-seg.git
cd point-supervised-building-seg
pip install -r requirements.txt
```

---

## Dataset

Download **SpaceNet-7** from Kaggle:

```
https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development
```

Place it so the path looks like:
```
data/
└── spacenet-7-multitemporal-urban-development/
    └── SN7_buildings_train/
        └── train/
            ├── L15-0331E-1257N_1327_3160_13/
            │   ├── images_masked/
            │   └── labels_match_pix/
            └── ...
```

---

## Training

Run all experiments:

```bash
python src/main.py --data_dir data/spacenet-7-multitemporal-urban-development/SN7_buildings_train/train
```

Skip one experiment:

```bash
# Only run sampling strategy comparison
python src/main.py --data_dir /path/to/train --skip_exp1

# Only run ratio ablation
python src/main.py --data_dir /path/to/train --skip_exp2
```

Custom epochs:

```bash
python src/main.py --data_dir /path/to/train --epochs 10
```

---

## Method

### Partial Cross-Entropy Loss

Standard cross-entropy computes gradients for every pixel. When most pixels are unlabeled, this introduces noise. Partial CE only computes loss on labeled pixels:

$$\mathcal{L}_{pCE} = \frac{\sum_i \mathbf{1}(y_i \neq 255) \cdot CE(p_i, y_i)}{\sum_i \mathbf{1}(y_i \neq 255)}$$

Unlabeled pixels are assigned the ignore index (255) and produce zero gradient.

### Sampling Strategies

**Random Sampling** — Pixels are selected uniformly at random across the entire image, giving the model a balanced view of all regions.

**Boundary-Aware Sampling** — Half the annotation budget is placed near building edges (detected via binary dilation), the other half on interior and background pixels.

---

## Hardware

Trained on 2× NVIDIA Tesla T4 (16 GB VRAM each) via Kaggle, using PyTorch `DataParallel`. Training takes ~20 GPU-minutes per 5-epoch run.

---

## License

[MIT](LICENSE)
