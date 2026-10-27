#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
============================================================================
 G13 :: Multi-Objective Genetic Algorithm for Lightweight CNN Design
 Lithography Hotspot Detection on the ICCAD-12 Benchmark (5 datasets)

 Research Question:
   Can a Genetic Algorithm simultaneously optimize hotspot-detection
   performance (validation Balanced Accuracy) AND CNN computational
   complexity (trainable-parameter count)?

 Author  : Group 13 (24BVD1033, 24BVD1036, 24BVD1041)
 Course  : BEVD402L - AI and Machine Learning for IC, VIT Chennai

 USAGE
 -----
     python g13_ga_cnn.py --data_root "C:\path\to\iccad-official" --output_dir results --mode debug
     python g13_ga_cnn.py --data_root "C:\path\to\iccad-official" --output_dir results --mode full

 Balanced-accuracy improvements in this revision:
 - GA candidate fitness uses validation Balanced Accuracy after validation-only threshold tuning;
 - thresholds are selected on validation data and frozen before test evaluation;
 - final GA architecture is chosen by a validation performance-constrained Pareto rule;
 - per-image test predictions saved for paired error analysis;
 - expanded full-mode GA search and training defaults;
 - mean MAC/FLOP estimates are populated in the overall summary.
Within each run, the official test split is excluded from model selection and used only for final evaluation.
Prior exploratory runs may already have exposed test metrics; therefore this script does not claim project-wide test-set blindness.
============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import logging
import platform
import traceback
import hashlib
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Headless plotting backend MUST be set before pyplot import
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, UnidentifiedImageError

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================================
# 0. GLOBAL CONSTANTS / LOGGING
# ============================================================================

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
BENCHMARK_DISPLAY = {b: f"ICCAD{b[-1]}" for b in BENCHMARKS}

LOGGER = logging.getLogger("g13_ga_cnn")


def setup_logger(log_path: Path) -> None:
    """Configure root logger to write to both console and file."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    LOGGER.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def set_global_seed(seed: int) -> None:
    """Fix seeds for python, numpy and tensorflow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class Config:
    data_root: str
    output_dir: str = "results"
    seed: int = 42

    # ---- image preprocessing ----
    # Images are resized to image_size and scaled to [0, 1]. The default remains 64x64;
    # alternate resolutions can be requested explicitly with --image_size.
    # Parameter counts are then calculated from the instantiated model rather
    # than copied from a paper or hardcoded.
    image_size: Tuple[int, int] = (64, 64)
    grayscale: bool = True                   # auto-detected from images

    # ---- data split ----
    val_ratio: float = 0.20

    # ---- baseline training ----
    baseline_epochs: Optional[int] = None    # None = use mode default
    baseline_batch_size: int = 32
    baseline_lr: float = 2e-3
    early_stopping_patience: int = 5

    # ---- GA search ----
    mode: str = "debug"          # "debug" or "full"
    population_size: Optional[int] = None
    generations: Optional[int] = None
    search_epochs: Optional[int] = None
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    ga_batch_size: int = 32

    # ---- final model ----
    final_epochs: Optional[int] = None       # None = use mode default; 0 = intentional debug skip

    # ---- objectives ----
    n_objectives: int = 2        # 2 = (BalAcc, params). Kept as genuine 2-objective NSGA-II.

    # ---- misc ----
    max_images_per_class: Optional[int] = None   # cap for debug speed; None = no cap
    inference_timing_repeats: int = 100
    overwrite: bool = False

    def apply_mode_defaults(self) -> None:
        """Fill in any field left as None with a mode-specific default.
        Explicitly-provided values (including 0) are preserved and only
        capped in debug mode to keep smoke tests fast -- they are never
        silently reset to 0. This replaces a previous buggy 'sentinel=0'
        pattern that could unintentionally skip final training."""
        if self.mode == "debug":
            self.population_size = 4 if self.population_size is None else self.population_size
            self.generations = 2 if self.generations is None else self.generations
            self.search_epochs = 1 if self.search_epochs is None else self.search_epochs
            self.baseline_epochs = 3 if self.baseline_epochs is None else min(self.baseline_epochs, 5)
            # final_epochs: default to 3 if unspecified. If the user explicitly
            # passed 0, that is honoured (intentional zero-epoch smoke test) and
            # is clearly labeled downstream rather than silently proceeding.
            self.final_epochs = 3 if self.final_epochs is None else min(self.final_epochs, 5)
            self.max_images_per_class = 150 if self.max_images_per_class is None else self.max_images_per_class
        else:  # full
            self.population_size = 24 if self.population_size is None else self.population_size
            self.generations = 12 if self.generations is None else self.generations
            self.search_epochs = 8 if self.search_epochs is None else self.search_epochs
            self.baseline_epochs = 25 if self.baseline_epochs is None else self.baseline_epochs
            self.final_epochs = 30 if self.final_epochs is None else self.final_epochs
            # max_images_per_class stays None (unlimited) unless user set it


def build_config_from_args() -> Config:
    p = argparse.ArgumentParser(description="G13 Multi-Objective GA for Lightweight CNN (ICCAD-12)")
    p.add_argument("--data_root", type=str, required=True,
                    help=r'Path to iccad-official folder, e.g. "C:\path\to\iccad-official"')
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="debug", choices=["debug", "full"],
                    help="debug = fast smoke test, full = final experiment")
    p.add_argument("--population", type=int, default=None, help="GA population size")
    p.add_argument("--generations", type=int, default=None, help="GA generations")
    p.add_argument("--search_epochs", type=int, default=None, help="epochs per GA candidate")
    p.add_argument("--final_epochs", type=int, default=None,
                    help="epochs for final GA retrain. 0 is honoured as an explicit "
                         "debug smoke-test request (skips real final training) and is clearly labeled.")
    p.add_argument("--baseline_epochs", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=0.20)
    p.add_argument("--image_size", type=int, default=64, help="square image side, e.g. 64")
    p.add_argument("--n_objectives", type=int, default=2, choices=[2],
                    help="Kept as 2 (BalAcc, params) for this version -- see chat notes.")
    p.add_argument("--max_images_per_class", type=int, default=None,
                    help="cap images/class (speed). Default: 150 in debug, unlimited in full")
    p.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting an existing non-empty output_dir. "
                         "Without this flag, a timestamped sibling directory is used instead "
                         "so previous debug/full runs are never silently overwritten.")
    args = p.parse_args()

    cfg = Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        mode=args.mode,
        image_size=(args.image_size, args.image_size),
        val_ratio=args.val_ratio,
        n_objectives=args.n_objectives,
        population_size=args.population,
        generations=args.generations,
        search_epochs=args.search_epochs,
        final_epochs=args.final_epochs,
        baseline_epochs=args.baseline_epochs,
        max_images_per_class=args.max_images_per_class,
        overwrite=args.overwrite,
    )
    if args.image_size <= 0:
        p.error("--image_size must be a positive integer")
    if not 0.0 < args.val_ratio < 1.0:
        p.error("--val_ratio must be strictly between 0 and 1")
    if args.max_images_per_class is not None and args.max_images_per_class <= 0:
        p.error("--max_images_per_class must be a positive integer when supplied")
    for name in ("population", "generations", "search_epochs", "baseline_epochs", "final_epochs"):
        value = getattr(args, name)
        if value is not None and value < 0:
            p.error(f"--{name} cannot be negative")
    cfg.apply_mode_defaults()
    return cfg


def resolve_output_dir(base_output_dir: Path, mode: str, overwrite: bool) -> Path:
    """Never silently overwrite a previous run's results. If base_output_dir
    already exists and has content, and --overwrite was not passed, a new
    timestamped sibling directory is used instead."""
    if overwrite or (not base_output_dir.exists()):
        return base_output_dir
    try:
        has_content = any(base_output_dir.iterdir())
    except FileNotFoundError:
        has_content = False
    if not has_content:
        return base_output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = base_output_dir.parent / f"{base_output_dir.name}_{mode}_{timestamp}"
    print(f"[NOTICE] '{base_output_dir}' already contains results and --overwrite was not passed. "
          f"Writing this run's output to a new directory instead: '{new_dir}'")
    return new_dir


# ============================================================================
# 2. DATASET DISCOVERY / LOADING
# ============================================================================

def discover_dataset(data_root: Path, benchmark: str) -> Dict[str, Path]:
    """Locate the train_hs/train_nhs/test_hs/test_nhs folders for a benchmark.
    Raises FileNotFoundError with a clear message if the expected layout
    is missing."""
    base = data_root / benchmark
    paths = {
        "train_hs": base / "train" / "train_hs",
        "train_nhs": base / "train" / "train_nhs",
        "test_hs": base / "test" / "test_hs",
        "test_nhs": base / "test" / "test_nhs",
    }
    missing = [str(v) for v in paths.values() if not v.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"[{benchmark}] Missing expected folder(s): {missing}. "
            f"Expected structure: {base}/train/train_hs, train_nhs, "
            f"{base}/test/test_hs, test_nhs"
        )
    return paths


def _list_image_files(folder: Path) -> List[Path]:
    return sorted([f for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS])


def inspect_image_properties(paths: Dict[str, Path], n_samples: int = 25) -> Dict[str, Any]:
    """Inspect a handful of real images to determine size/mode/pixel range."""
    sample_files = []
    for key in ["train_hs", "train_nhs"]:
        files = _list_image_files(paths[key])
        sample_files.extend(files[:n_samples])
    if not sample_files:
        raise RuntimeError("No image files found to inspect image properties.")

    sizes, modes, pix_min, pix_max = [], [], [], []
    for f in sample_files:
        try:
            with Image.open(f) as im:
                sizes.append(im.size)
                modes.append(im.mode)
                arr = np.asarray(im.convert("L"))
                pix_min.append(int(arr.min()))
                pix_max.append(int(arr.max()))
        except (UnidentifiedImageError, OSError) as e:
            LOGGER.warning(f"Could not inspect image {f}: {e}")

    most_common_mode = max(set(modes), key=modes.count) if modes else "L"
    most_common_size = max(set(sizes), key=sizes.count) if sizes else None
    return {
        "n_inspected": len(sample_files),
        "unique_sizes": sorted(set(sizes)),
        "most_common_size": most_common_size,
        "most_common_mode": most_common_mode,
        "pixel_min": min(pix_min) if pix_min else None,
        "pixel_max": max(pix_max) if pix_max else None,
    }


def check_input_resolution(img_props: Dict[str, Any], cfg: Config, benchmark: str) -> Dict[str, Any]:
    """Explicitly compare the DETECTED native image resolution against the
    CONFIGURED target resolution (cfg.image_size) and surface the finding
    rather than silently resizing without comment. 64x64 is retained per
    explicit instruction; no authoritative source in this project justifies
    a different value."""
    native = img_props.get("most_common_size")
    target = cfg.image_size
    note = {
        "benchmark": benchmark,
        "native_resolution_detected": native,
        "configured_target_resolution": target,
        "resolution_source": (
            f"Configured project input is {target[0]}x{target[1]} pixels, with 64x64 as the "
            "default target requested for this project; this is the actual preprocessing target "
            "and is not a claim about the resolution used by any prior paper."
        ),
        "configured_pixel_count": int(target[0] * target[1]),
        "preprocessing": f"Resize to configured {target[0]}x{target[1]} target, convert to grayscale/RGB, scale uint8 pixels to float32 [0,1].",
    }
    if native is not None:
        nx, ny = native
        tx, ty = target
        downsample_x = nx / tx if tx else None
        downsample_y = ny / ty if ty else None
        pixel_ratio = (nx * ny) / (tx * ty) if (tx and ty) else None
        note.update({
            "downsample_factor_x": round(downsample_x, 2) if downsample_x else None,
            "downsample_factor_y": round(downsample_y, 2) if downsample_y else None,
            "pixel_count_reduction_factor": round(pixel_ratio, 1) if pixel_ratio else None,
        })
        if downsample_x and downsample_x > 4:
            LOGGER.warning(
                f"[{benchmark}] RESOLUTION CHECK: native images are ~{native}, "
                f"configured target is {target} -> downsampling by ~{downsample_x:.1f}x per axis "
                f"(~{pixel_ratio:.0f}x fewer pixels). The configured target {target} is being kept "
                "as-is (not silently changed)."
            )
        else:
            LOGGER.info(f"[{benchmark}] RESOLUTION CHECK: native {native} vs target {target} -- "
                        f"downsampling factor ~{downsample_x:.2f}x per axis.")
    else:
        LOGGER.warning(f"[{benchmark}] RESOLUTION CHECK: could not determine native image size; "
                        f"proceeding with configured target {target}.")
    return note


def load_images(folder: Path, label: int, image_size: Tuple[int, int],
                 grayscale: bool, max_images: Optional[int],
                 corrupt_log: List[str], seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load, resize and normalize images from `folder`. If capped, choose a
    deterministic random subset instead of the first alphabetically sorted
    files. A cap applies to both train and test folders and must be reported
    as a subset evaluation, not a full official-test evaluation."""
    files = _list_image_files(folder)
    if max_images is not None and len(files) > max_images:
        # Stable folder-derived seed: independent of Python's randomized hash().
        folder_salt = int(hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng((int(seed) + folder_salt) % (2**32))
        chosen = np.sort(rng.choice(len(files), size=max_images, replace=False))
        files = [files[int(i)] for i in chosen]

    images, labels = [], []
    mode = "L" if grayscale else "RGB"
    for f in files:
        try:
            with Image.open(f) as im:
                # Explicit resize to the configured target (64x64 by default) is the pixel-resolution correction.
                # Use the modern Pillow enum where available, with a compatibility fallback.
                resampling = getattr(Image, "Resampling", Image).BILINEAR
                im = im.convert(mode).resize(image_size, resampling)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                if grayscale:
                    arr = arr[..., np.newaxis]
                images.append(arr)
                labels.append(label)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            msg = f"CORRUPT/UNREADABLE: {f} -> {e}"
            LOGGER.warning(msg)
            corrupt_log.append(msg)

    if not images:
        raise RuntimeError(f"No valid images could be loaded from {folder}")

    channels = 1 if grayscale else 3
    X = np.stack(images).astype(np.float32).reshape(-1, image_size[0], image_size[1], channels)
    y = np.array(labels, dtype=np.int32)
    return X, y


# ============================================================================
# 3. DATASET ANALYSIS
# ============================================================================

def analyze_dataset(paths: Dict[str, Path], benchmark: str) -> Dict[str, Any]:
    """Count class sizes and compute imbalance statistics (directory-listing
    based, so this is cheap and covers the FULL official dataset)."""
    counts = {k: len(_list_image_files(v)) for k, v in paths.items()}
    train_total = counts["train_hs"] + counts["train_nhs"]
    test_total = counts["test_hs"] + counts["test_nhs"]

    hs_pct_train = 100.0 * counts["train_hs"] / train_total if train_total else 0.0
    nhs_pct_train = 100.0 - hs_pct_train
    imbalance_ratio = (counts["train_nhs"] / counts["train_hs"]) if counts["train_hs"] else float("inf")

    stats = {
        "benchmark": benchmark,
        "train_hs": counts["train_hs"],
        "train_nhs": counts["train_nhs"],
        "test_hs": counts["test_hs"],
        "test_nhs": counts["test_nhs"],
        "train_total": train_total,
        "test_total": test_total,
        "hs_pct_train": round(hs_pct_train, 3),
        "nhs_pct_train": round(nhs_pct_train, 3),
        "imbalance_ratio_nhs_per_hs": round(imbalance_ratio, 3),
    }
    return stats


def plot_class_distribution(stats: Dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, split in zip(axes, ["train", "test"]):
        hs = stats[f"{split}_hs"]
        nhs = stats[f"{split}_nhs"]
        bars = ax.bar(["Hotspot (HS)", "Non-Hotspot (NHS)"], [hs, nhs],
                       color=["#d62728", "#1f77b4"])
        ax.set_title(f"{BENCHMARK_DISPLAY[stats['benchmark']]} - {split.capitalize()} Split")
        ax.set_ylabel("Number of samples")
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{int(h)}", (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Class Distribution -- {BENCHMARK_DISPLAY[stats['benchmark']]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================================

def split_training_data(X: np.ndarray, y: np.ndarray, val_ratio: float,
                         seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of the OFFICIAL training data only. The official
    test set is never touched here."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_ratio, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def preprocess_images(X: np.ndarray) -> np.ndarray:
    """Identity preprocessing hook (images are already resized/normalized
    to [0,1] in load_images). Kept as an explicit named step so any future
    shared preprocessing logic has one place to live."""
    return X.astype(np.float32)


# ============================================================================
# 5. MODEL BUILDERS
# ============================================================================

def basic_block(x, filters: int, kernel: int, name_prefix: str):
    """One 'Basic Block' as defined in the reference paper:
    Conv-ELU, Conv-ELU, Conv(linear) -> BN -> ELU -> MaxPool(2x2)."""
    x = layers.Conv2D(filters, kernel, padding="same", activation="elu",
                       name=f"{name_prefix}_conv1")(x)
    x = layers.Conv2D(filters, kernel, padding="same", activation="elu",
                       name=f"{name_prefix}_conv2")(x)
    x = layers.Conv2D(filters, kernel, padding="same", activation=None,
                       name=f"{name_prefix}_conv3")(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3,
                                   name=f"{name_prefix}_bn")(x)
    x = layers.ELU(name=f"{name_prefix}_elu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name_prefix}_pool")(x)
    return x


def build_hand_designed_lightweight_model(input_shape: Tuple[int, int, int], learning_rate: float) -> keras.Model:
    """Independent hand-designed compact CNN comparator (not GA searched).

    Deliberately uses a small separable-convolution stack as a practical
    lightweight baseline. Parameter count and latency are measured from the
    instantiated model, not assumed.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.SeparableConv2D(8, 3, padding="same", activation="relu", name="hand_sepconv1")(inputs)
    x = layers.MaxPooling2D(2, name="hand_pool1")(x)
    x = layers.SeparableConv2D(12, 3, padding="same", activation="relu", name="hand_sepconv2")(x)
    x = layers.MaxPooling2D(2, name="hand_pool2")(x)
    x = layers.GlobalAveragePooling2D(name="hand_global_average_pool")(x)
    x = layers.Dropout(0.25, name="hand_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    model = keras.Model(inputs, outputs, name="hand_designed_separable_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_baseline_model(input_shape: Tuple[int, int, int], learning_rate: float) -> keras.Model:
    """Reference-paper lightweight CNN, UNCHANGED:
    Input -> BasicBlock -> 5x5 MaxPool -> BasicBlock -> Flatten
           -> Dropout(0.3) -> Dense(sigmoid)
    Filters/kernel/dropout are kept IDENTICAL to the paper; only the Dense
    layer's input size follows from the actual image resolution. Parameter
    count is reported directly from model.summary(), never assumed."""
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = basic_block(inputs, filters=12, kernel=3, name_prefix="block1")
    x = layers.MaxPooling2D(pool_size=(5, 5), padding="same", name="pool_5x5")(x)
    x = basic_block(x, filters=12, kernel=3, name_prefix="block2")
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="baseline_lightweight_cnn")
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


@dataclass
class Chromosome:
    """GA-optimizable CNN design. Fixed/non-optimized items (optimizer type,
    activation choice, BN hyper-params, loss function, the fixed 5x5 pool
    position) are documented here and are NOT part of this chromosome."""
    depth: int          # number of basic blocks: 1, 2, 3  (never 0)
    filters: int         # filters per conv layer: 8,12,16,24,32
    kernel: int          # kernel size: 3 or 5
    dense_units: int     # 8,16,32,64
    dropout: float        # 0.1 - 0.5
    learning_rate: float  # 1e-4 - 1e-2

    def key(self) -> str:
        """Stable string key used for duplicate-candidate detection/caching."""
        raw = f"{self.depth}|{self.filters}|{self.kernel}|{self.dense_units}|{round(self.dropout,3)}|{round(self.learning_rate,6)}"
        return hashlib.md5(raw.encode()).hexdigest()


GENE_SPACE = {
    "depth": [1, 2, 3],
    "filters": [8, 12, 16, 24, 32],
    "kernel": [3, 5],
    "dense_units": [8, 16, 32, 64],
}
DROPOUT_RANGE = (0.1, 0.5)
LR_RANGE = (1e-4, 1e-2)

CANDIDATE_ARCHITECTURE_DEFINITION = """
GA CANDIDATE ARCHITECTURE FAMILY (mirrors the baseline's structural design)
----------------------------------------------------------------------------
Input
  -> Basic Block #1            (filters, kernel: GA-controlled)
  -> 5x5 MaxPool                (FIXED position, identical to baseline, padding='same')
  -> Basic Block #2             (only if depth > 1; same filters/kernel as block #1)
  -> Basic Block #3             (only if depth > 2; same filters/kernel as block #1)
  -> Flatten
  -> Dropout(rate)               (GA-controlled)
  -> Dense(dense_units, ELU)    (GA-controlled)
  -> Dense(1, Sigmoid)

A Basic Block = Conv2D(elu) -> Conv2D(elu) -> Conv2D(linear) -> BatchNorm -> ELU -> MaxPool(2x2)
(identical internal structure to the baseline's Basic Block; only filters/kernel vary by GA gene)

depth in {1,2,3} -- depth=0 is structurally impossible (GENE_SPACE excludes it).
The 5x5 MaxPool is FIXED after the first Basic Block, exactly as in the baseline,
so the GA optimizes depth/filters/kernel/dense_units/dropout/learning_rate while
keeping the same early spatial-reduction strategy as the reference architecture.
""".strip()

GA_OPERATOR_DEFINITION = """
SEARCH SPACE AND OPERATORS
--------------------------
Genes: depth {1,2,3}; filters {8,12,16,24,32}; kernel {3,5};
       dense units {8,16,32,64}; dropout [0.1,0.5]; learning rate [1e-4,1e-2].
Algorithm: NSGA-II, two objectives (maximize validation balanced accuracy at
fixed threshold 0.5; minimize trainable parameter count).
Selection: NSGA-II non-dominated sorting and crowding-distance selection.
Variation: DEAP varAnd with simulated-binary bounded crossover
(cxSimulatedBinaryBounded, eta=15) and polynomial bounded mutation
(mutPolynomialBounded, eta=20, per-gene probability 1/6); crossover
probability 0.7 and mutation probability 0.3 by default.
The final selected model's decision threshold is separately selected on validation
data only and then frozen for test evaluation.
""".strip()


def decode_chromosome(genes: List[float]) -> Chromosome:
    """Decode a real-valued GA individual (as produced by DEAP) into a
    concrete, bounded, discretised Chromosome."""
    def snap(val, choices):
        idx = int(round(val))
        idx = max(0, min(idx, len(choices) - 1))
        return choices[idx]

    depth = snap(genes[0], GENE_SPACE["depth"])
    filters = snap(genes[1], GENE_SPACE["filters"])
    kernel = snap(genes[2], GENE_SPACE["kernel"])
    dense_units = snap(genes[3], GENE_SPACE["dense_units"])
    dropout = float(np.clip(genes[4], *DROPOUT_RANGE))
    lr = float(np.clip(genes[5], *LR_RANGE))
    return Chromosome(depth, filters, kernel, dense_units, dropout, lr)


def build_candidate_model(chromosome: Chromosome, input_shape: Tuple[int, int, int]) -> keras.Model:
    """Build a CNN from a GA chromosome, using the SAME structural family as
    the baseline: Basic Block #1 -> fixed 5x5 MaxPool -> optional further
    Basic Blocks (depth>1) -> Flatten -> Dropout -> Dense -> Sigmoid.
    This keeps GA candidates meaningfully comparable to the baseline and
    avoids uncontrolled parameter blow-up from a missing early pooling stage."""
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = basic_block(inputs, filters=chromosome.filters, kernel=chromosome.kernel,
                     name_prefix="cand_block1")
    x = layers.MaxPooling2D(pool_size=(5, 5), padding="same", name="cand_pool_5x5")(x)

    for i in range(2, chromosome.depth + 1):
        h = x.shape[1] or 1
        w = x.shape[2] or 1
        if min(h, w) < 2:
            LOGGER.warning(f"Candidate depth truncated before block {i}: "
                            f"feature map already too small ({h}x{w}) to pool further.")
            break
        x = basic_block(x, filters=chromosome.filters, kernel=chromosome.kernel,
                         name_prefix=f"cand_block{i}")

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(chromosome.dropout, name="dropout")(x)
    x = layers.Dense(chromosome.dense_units, activation="elu", name="dense_hidden")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="ga_candidate_cnn")
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=chromosome.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================================
# 6. TRAINING / METRICS / COMPLEXITY
# ============================================================================

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_pos = max(n_pos, 1)
    n_neg = max(n_neg, 1)
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def train_model(model: keras.Model, X_tr, y_tr, X_val, y_val,
                 epochs: Optional[int], batch_size: int, patience: int = 5,
                 verbose: int = 0, label: str = "model") -> Optional[keras.callbacks.History]:
    """Trains model for `epochs` epochs. If epochs is None or < 1, training
    is explicitly SKIPPED (model remains at random initialization) and a
    loud warning is logged -- callers MUST check the return value and tag
    any resulting metrics as a debug/smoke-test artifact, never as a real
    result. Training uses X_tr/y_tr only; X_val/y_val are used ONLY for
    val_loss monitoring / EarlyStopping, never for gradient updates. The
    official test set is never passed to this function at all."""
    if epochs is None or epochs < 1:
        LOGGER.warning(
            f"[{label}] epochs={epochs} -> TRAINING SKIPPED. Model remains at random "
            f"initialization. Any metrics computed from this model are a debug/smoke-test "
            f"artifact only and must NOT be reported as a real experimental result."
        )
        return None

    class_weight = compute_class_weights(y_tr)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True)
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity = TN / (TN + FP), with zero-division protection."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Positive = Hotspot(1), Negative = Non-Hotspot(0).
    Balanced Accuracy = (Sensitivity + Specificity) / 2, computed explicitly
    (and cross-checked against sklearn's own balanced_accuracy_score)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # = sensitivity = TPR
    specificity = calculate_specificity(y_true, y_pred)      # = TNR
    bal_acc = (recall + specificity) / 2.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sk_bal_acc = balanced_accuracy_score(y_true, y_pred)
    if abs(sk_bal_acc - bal_acc) > 1e-6:
        LOGGER.warning(f"Balanced-accuracy mismatch: manual={bal_acc:.6f} sklearn={sk_bal_acc:.6f}")
    return {
        "accuracy": acc, "balanced_accuracy": bal_acc, "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def validation_multi_metric_score(metrics: Dict[str, Any]) -> float:
    """Equal-weight validation score for threshold/model selection.

    The score is the mean of Balanced Accuracy, precision, recall and F1.
    It is a model-selection criterion, not a test metric and not a guarantee
    that every metric will beat the baseline.
    """
    return float(np.mean([metrics["balanced_accuracy"], metrics["precision"],
                          metrics["recall"], metrics["f1"]]))


def choose_balanced_accuracy_threshold(y_true: np.ndarray,
                                       probabilities: np.ndarray) -> Tuple[float, float]:
    """Choose the threshold that maximizes VALIDATION balanced accuracy only.

    This aligns threshold selection with the project's primary objective.
    Ties are broken by choosing the threshold closest to 0.5. The official
    test labels are never passed to this function. Returns (threshold, val BA).
    """
    thresholds = np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), np.array([0.5])]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(y_true, predictions)
        score = metrics["balanced_accuracy"]
        if (score > best_score + 1e-12 or
                (abs(score - best_score) <= 1e-12 and
                 abs(float(threshold) - 0.5) < abs(best_threshold - 0.5))):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def calculate_complexity(model: keras.Model) -> Dict[str, Any]:
    """Report exact Keras parameter counts and a clearly labelled operation estimate.

    Parameter counts are read from the instantiated model. MACs are estimated
    for Conv2D and Dense layers from their symbolic input/output shapes. FLOPs
    are approximated as 2 * MACs (one multiply plus one add per MAC). This is
    NOT a complete hardware FLOP count: bias additions, BatchNorm, pooling,
    activations, memory movement and framework/hardware effects are excluded.
    The primary complexity measure for this project is trainable parameters;
    MACs/FLOPs and measured inference latency are supplementary metrics.
    """
    trainable = int(sum(np.prod(tuple(v.shape)) for v in model.trainable_weights))
    non_trainable = int(sum(np.prod(tuple(v.shape)) for v in model.non_trainable_weights))
    total = trainable + non_trainable

    macs = 0
    flops_available = True
    included_layers = []
    try:
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                in_shape = tuple(layer.input.shape)
                out_shape = tuple(layer.output.shape)
                # The batch dimension may be None; spatial/channel dimensions must be static.
                if len(in_shape) != 4 or len(out_shape) != 4:
                    raise ValueError(f"Unexpected Conv2D tensor rank for layer {layer.name}")
                _, in_h, in_w, in_channels = in_shape
                _, out_h, out_w, out_channels = out_shape
                dims = (in_channels, out_h, out_w, out_channels)
                if any(d is None for d in dims):
                    raise ValueError(f"Dynamic Conv2D dimensions for layer {layer.name}: {dims}")
                kh, kw = layer.kernel_size
                groups = int(getattr(layer, "groups", 1))
                if int(in_channels) % groups != 0:
                    raise ValueError(f"Input channels not divisible by groups for {layer.name}")
                macs += int(out_h) * int(out_w) * int(out_channels) * (
                    int(kh) * int(kw) * (int(in_channels) // groups)
                )
                included_layers.append(layer.name)

            elif isinstance(layer, layers.Dense):
                in_shape = tuple(layer.input.shape)
                out_shape = tuple(layer.output.shape)
                if len(in_shape) < 2 or len(out_shape) < 2:
                    raise ValueError(f"Unexpected Dense tensor rank for layer {layer.name}")
                if any(d is None for d in in_shape[1:]) or out_shape[-1] is None:
                    raise ValueError(f"Dynamic Dense dimensions for layer {layer.name}")
                # Dense can operate on rank >2 tensors; count every vector position.
                positions = int(np.prod(in_shape[1:-1])) if len(in_shape) > 2 else 1
                macs += positions * int(in_shape[-1]) * int(out_shape[-1])
                included_layers.append(layer.name)
    except Exception as e:
        LOGGER.warning(f"MACs/FLOPs estimate unavailable for model {model.name}: {e}")
        flops_available = False
        macs = None

    flops = (2 * int(macs)) if (flops_available and macs is not None) else None
    note = (
        "Estimated Conv2D+Dense MACs only; estimated FLOPs = 2 x MACs. "
        "Excludes bias additions, BatchNorm, pooling, activations, memory movement, "
        "and framework/hardware overhead. Not an exact hardware execution count. "
        "Trainable parameters are the primary complexity measure; inference latency "
        "is measured separately."
        if flops_available else
        "Parameter counts are exact from Keras weights. MACs/FLOPs estimate unavailable "
        "because one or more required layer dimensions could not be resolved. "
        "Trainable parameters remain the primary complexity measure."
    )

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "estimated_macs": int(macs) if macs is not None else None,
        "estimated_flops": flops,
        "complexity_primary_metric": "trainable_parameters",
        "complexity_layers_included": ",".join(included_layers),
        "flops_measurement_note": note,
    }


def describe_complexity_report(complexity: Dict[str, Any], model_label: str) -> str:
    """Create a report-safe complexity statement without hardcoded FLOPs claims."""
    macs = complexity.get("estimated_macs")
    flops = complexity.get("estimated_flops")
    macs_text = f"{macs:,}" if macs is not None else "unavailable"
    flops_text = f"{flops:,}" if flops is not None else "unavailable"
    return (
        f"{model_label}: {complexity['trainable_params']:,} trainable parameters; "
        f"{complexity['total_params']:,} total parameters (including non-trainable weights). "
        f"Primary complexity metric: trainable parameter count. Supplementary operation "
        f"estimates: {macs_text} Conv2D/Dense MACs and {flops_text} estimated FLOPs "
        f"(FLOPs approximated as 2 x MACs where available). These operation counts are "
        f"partial estimates, not exact hardware FLOP counts. Inference latency is measured "
        f"separately on the test images after model selection.\n"
        f"Measurement note: {complexity.get('flops_measurement_note', 'Not available.')}"
    )


def measure_inference_time(model: keras.Model, X_sample: np.ndarray,
                            n_repeats: int = 100, batch_size: int = 1) -> Dict[str, float]:
    """Consistent single-image inference timing: warm-up, then repeated
    timed passes over a fixed sample, batch_size=1, mean +/- std per image
    (in milliseconds)."""
    n = min(len(X_sample), 100)
    if n < 1:
        raise ValueError("Cannot measure inference latency: X_sample contains no images.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    X_sample = X_sample[:n]

    _ = model.predict(X_sample[:min(8, n)], batch_size=batch_size, verbose=0)  # warm-up

    per_image_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = model.predict(X_sample, batch_size=batch_size, verbose=0)
        elapsed = time.perf_counter() - start
        per_image_times.append((elapsed / n) * 1000.0)  # ms/image

    return {
        "inference_time_ms_mean": float(np.mean(per_image_times)),
        "inference_time_ms_std": float(np.std(per_image_times)),
        "n_timing_repeats": n_repeats,
        "n_timing_images": n,
    }


# ============================================================================
# 7. GENETIC ALGORITHM (genuine two-objective NSGA-II via DEAP)
# ============================================================================

def _clear_keras_session():
    keras.backend.clear_session()


class GAEvaluationCache:
    """Caches fitness results for identical chromosomes within one run so
    the GA never retrains a duplicate candidate."""
    def __init__(self):
        self._store: Dict[str, Tuple[float, ...]] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: Tuple[float, ...]):
        self._store[key] = value


def evaluate_candidate(genes, input_shape, X_tr, y_tr, X_val, y_val,
                        search_epochs, batch_size, n_objectives,
                        cache: GAEvaluationCache, records: List[Dict[str, Any]],
                        benchmark: str, generation: int, candidate_counter: List[int]
                        ) -> Tuple[float, ...]:
    """Full candidate-evaluation pipeline: decode -> build -> train
    (on TRAIN split only) -> evaluate (on VALIDATION split only) -> return
    the NSGA-II fitness tuple (maximize BalAcc, minimize trainable params).
    Never touches the test set. Any exception is caught so a single failed
    candidate cannot kill the whole GA run; it is penalised with the worst
    possible fitness and logged with status='FAILED' instead."""
    chromosome = decode_chromosome(genes)
    ckey = chromosome.key()
    cached = cache.get(ckey)
    candidate_counter[0] += 1
    cid = candidate_counter[0]

    if cached is not None:
        LOGGER.info(f"  [{benchmark}] gen{generation} cand{cid}: duplicate chromosome -> using cached fitness")
        return cached

    fitness = None
    try:
        _clear_keras_session()
        model = build_candidate_model(chromosome, input_shape)
        history = train_model(model, X_tr, y_tr, X_val, y_val, epochs=search_epochs,
                               batch_size=batch_size, patience=3, verbose=0,
                               label=f"{benchmark}-gen{generation}-cand{cid}")
        trained = history is not None

        y_val_prob = model.predict(X_val, batch_size=batch_size, verbose=0).ravel()
        # Use validation-only threshold tuning for candidate fitness so the GA
        # optimizes the same primary metric used for final model evaluation.
        # The official test set is never used to choose this threshold.
        search_threshold, tuned_val_bal_acc = choose_balanced_accuracy_threshold(y_val, y_val_prob)
        y_val_pred = (y_val_prob >= search_threshold).astype(int)
        metrics = calculate_metrics(y_val, y_val_pred)
        # Guard against accidental mismatch between the selected threshold score
        # and the metrics computed from its predictions.
        bal_acc_from_predictions = metrics["balanced_accuracy"]
        if abs(bal_acc_from_predictions - tuned_val_bal_acc) > 1e-9:
            raise RuntimeError("Validation threshold score and calculated balanced accuracy disagree")
        complexity = calculate_complexity(model)

        bal_acc = metrics["balanced_accuracy"]
        params = complexity["trainable_params"]
        fitness = (bal_acc, -params)  # DEAP maximizes both -> negate the minimize objective

        status = "ok" if trained else "SKIPPED_ZERO_EPOCHS_DEBUG"

        records.append({
            "benchmark": benchmark, "candidate_id": cid, "generation": generation,
            "depth": chromosome.depth, "filters": chromosome.filters,
            "kernel_size": chromosome.kernel, "dense_units": chromosome.dense_units,
            "dropout": chromosome.dropout, "learning_rate": chromosome.learning_rate,
            "accuracy": metrics["accuracy"], "balanced_accuracy": bal_acc,
            "validation_multi_metric_score": validation_multi_metric_score(metrics),
            "decision_threshold": search_threshold,
            "search_threshold_policy": "validation_tuned_balanced_accuracy_for_GA_fitness",
            "precision": metrics["precision"], "recall": metrics["recall"],
            "specificity": metrics["specificity"], "f1": metrics["f1"],
            "tp": metrics["TP"], "tn": metrics["TN"], "fp": metrics["FP"], "fn": metrics["FN"],
            "total_parameters": complexity["total_params"],
            "trainable_parameters": complexity["trainable_params"],
            "MACs": complexity["estimated_macs"], "FLOPs": complexity["estimated_flops"],
            "inference_time_ms": None,  # not timed during search to keep GA fast; see final model timing
            "architecture": json.dumps(asdict(chromosome)),
            "hyperparameters": json.dumps({"learning_rate": chromosome.learning_rate,
                                            "dropout": chromosome.dropout,
                                            "batch_size": batch_size,
                                            "search_epochs": search_epochs}),
            "pareto_optimal": False,  # filled in later
            "chromosome_key": ckey, "status": status, "error_message": None,
        })
        del model
        _clear_keras_session()

    except Exception as e:
        LOGGER.error(f"  [{benchmark}] gen{generation} cand{cid} FAILED: {e}\n{traceback.format_exc()}")
        fitness = (0.0, -1e12)
        records.append({
            "benchmark": benchmark, "candidate_id": cid, "generation": generation,
            "depth": chromosome.depth, "filters": chromosome.filters,
            "kernel_size": chromosome.kernel, "dense_units": chromosome.dense_units,
            "dropout": chromosome.dropout, "learning_rate": chromosome.learning_rate,
            "accuracy": None, "balanced_accuracy": 0.0, "decision_threshold": 0.5,
            "precision": None, "recall": None,
            "specificity": None, "f1": None, "tp": None, "tn": None, "fp": None, "fn": None,
            "total_parameters": None, "trainable_parameters": None,
            "MACs": None, "FLOPs": None, "inference_time_ms": None,
            "architecture": json.dumps(asdict(chromosome)),
            "hyperparameters": json.dumps({"learning_rate": chromosome.learning_rate,
                                            "dropout": chromosome.dropout}),
            "pareto_optimal": False, "chromosome_key": ckey,
            "status": "FAILED", "error_message": str(e), "FAILED": True,
        })

    cache.set(ckey, fitness)
    return fitness


def run_nsga2(input_shape, X_tr, y_tr, X_val, y_val, cfg: Config,
              benchmark: str) -> Tuple[List[Dict[str, Any]], List]:
    """Runs genuine multi-objective NSGA-II using DEAP over the Chromosome
    search space. This is NOT a weighted single-objective score -- DEAP's
    tools.selNSGA2 performs real non-dominated sorting + crowding distance."""
    from deap import base, creator, tools, algorithms

    weights = (1.0, 1.0)  # (maximize BalAcc, maximize -params == minimize params)

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    gene_bounds = [
        (0, len(GENE_SPACE["depth"]) - 1),
        (0, len(GENE_SPACE["filters"]) - 1),
        (0, len(GENE_SPACE["kernel"]) - 1),
        (0, len(GENE_SPACE["dense_units"]) - 1),
        DROPOUT_RANGE,
        LR_RANGE,
    ]

    def random_gene(i):
        lo, hi = gene_bounds[i]
        return random.uniform(lo, hi)

    def init_individual():
        return creator.Individual([random_gene(i) for i in range(len(gene_bounds))])

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    low = [b[0] for b in gene_bounds]
    up = [b[1] for b in gene_bounds]

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=20.0,
                      indpb=1.0 / len(gene_bounds))
    toolbox.register("select", tools.selNSGA2)

    records: List[Dict[str, Any]] = []
    cache = GAEvaluationCache()
    candidate_counter = [0]

    def eval_wrapper(individual, generation):
        return evaluate_candidate(
            list(individual), input_shape, X_tr, y_tr, X_val, y_val,
            cfg.search_epochs, cfg.ga_batch_size, cfg.n_objectives, cache, records,
            benchmark, generation, candidate_counter,
        )

    pop = toolbox.population(n=cfg.population_size)
    for ind in pop:
        ind.fitness.values = eval_wrapper(ind, generation=0)
    pop = toolbox.select(pop, cfg.population_size)

    convergence_log = []
    best_bal_acc_so_far = max(r["balanced_accuracy"] for r in records) if records else 0.0
    convergence_log.append({"generation": 0, "best_balanced_accuracy": best_bal_acc_so_far})

    for gen in range(1, cfg.generations + 1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cfg.crossover_prob, mutpb=cfg.mutation_prob)
        for ind in offspring:
            ind.fitness.values = eval_wrapper(ind, generation=gen)
        pop = toolbox.select(pop + offspring, cfg.population_size)

        gen_records = [r for r in records if r["generation"] <= gen]
        best_bal_acc_so_far = max(r["balanced_accuracy"] for r in gen_records) if gen_records else best_bal_acc_so_far
        convergence_log.append({"generation": gen, "best_balanced_accuracy": best_bal_acc_so_far})
        LOGGER.info(f"  [{benchmark}] Generation {gen}/{cfg.generations} complete. "
                    f"Best val BalAcc so far = {best_bal_acc_so_far:.4f}")

    return records, convergence_log


def identify_pareto_front(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Non-dominated sorting on (maximize balanced_accuracy, minimize
    trainable_parameters), computed over ALL valid evaluated candidates.
    Marks each record's 'pareto_optimal' flag and returns the Pareto-optimal
    subset. Failed/incomplete candidates are excluded from consideration."""
    valid = [r for r in records if r.get("trainable_parameters") is not None
             and r.get("status") != "FAILED"]
    pareto = []
    for i, a in enumerate(valid):
        dominated = False
        for j, b in enumerate(valid):
            if i == j:
                continue
            better_or_equal_acc = b["balanced_accuracy"] >= a["balanced_accuracy"]
            better_or_equal_params = b["trainable_parameters"] <= a["trainable_parameters"]
            strictly_better = (b["balanced_accuracy"] > a["balanced_accuracy"]) or \
                               (b["trainable_parameters"] < a["trainable_parameters"])
            if better_or_equal_acc and better_or_equal_params and strictly_better:
                dominated = True
                break
        a["pareto_optimal"] = not dominated
        if not dominated:
            pareto.append(a)
    return pareto


def calculate_pareto_hypervolume(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate normalized 2-D hypervolume for (maximize BA, minimize params).

    Parameter count is mapped to a normalized minimization coordinate using the
    min/max parameter counts among valid candidates. We maximize (BA, 1-norm_params)
    and use reference point (0.0, 0.0). This is a relative, per-benchmark indicator;
    compare hypervolume across runs only when search space and normalization policy match.
    """
    valid = [r for r in records if r.get("status") != "FAILED"
             and r.get("trainable_parameters") is not None
             and r.get("balanced_accuracy") is not None]
    if not valid:
        return {"hypervolume": 0.0, "reference_point": [0.0, 0.0],
                "normalization": "parameter count min-max normalized among valid candidates",
                "n_points": 0}
    pvals = [float(r["trainable_parameters"]) for r in valid]
    pmin, pmax = min(pvals), max(pvals)
    span = max(pmax - pmin, 1.0)
    points = []
    for r in valid:
        ba = float(np.clip(r["balanced_accuracy"], 0.0, 1.0))
        efficiency = 1.0 - float(np.clip((float(r["trainable_parameters"]) - pmin) / span, 0.0, 1.0))
        points.append((ba, efficiency))
    # Keep nondominated points, sort by increasing BA, and sum rectangles to origin.
    front = []
    for x, y in points:
        if not any((xx >= x and yy >= y and (xx > x or yy > y)) for xx, yy in points):
            front.append((x, y))
    front = sorted(set(front), key=lambda pt: pt[0])
    hv = 0.0
    prev_x = 0.0
    max_y = 0.0
    for x, y in front:
        if x > prev_x:
            hv += (x - prev_x) * max_y
            prev_x = x
        max_y = max(max_y, y)
    hv += max(0.0, 1.0 - prev_x) * max_y
    return {"hypervolume": float(hv), "reference_point": [0.0, 0.0],
            "normalization": "BA in [0,1]; efficiency=1-minmax(parameter count) among valid candidates",
            "n_points": len(front), "parameter_min": pmin, "parameter_max": pmax}


def select_operating_points(pareto: List[Dict[str, Any]], tolerance: float = 0.01
                             ) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Selects 3 practical points from the Pareto front, using VALIDATION
    metrics only (never test):
      A. Highest Balanced Accuracy
      B. Lowest trainable-parameter count
      C. Balanced trade-off (PERFORMANCE-CONSTRAINED RULE):
           STEP 1: Find the Pareto candidate with the maximum validation
                   Balanced Accuracy.
           STEP 2: Define PERFORMANCE_TOLERANCE = `tolerance` (default 0.01,
                   i.e. 1 percentage point of Balanced Accuracy).
           STEP 3: minimum_acceptable_balanced_accuracy =
                       maximum_validation_balanced_accuracy - tolerance
           STEP 4: Keep only Pareto candidates satisfying
                       candidate.validation_balanced_accuracy
                           >= minimum_acceptable_balanced_accuracy
           STEP 5: Among those acceptable candidates, select the one with
                   the LOWEST trainable parameter count.

    This is the CURRENT and ONLY selection rule for 'balanced_tradeoff'.
    It replaces an earlier normalized-Euclidean-distance-to-ideal-point
    rule (now fully removed), which could select a Pareto point that
    sacrificed a large amount of absolute validation accuracy simply
    because it looked "close" in a normalized (accuracy, params) space
    with a wide parameter range. Under this rule, performance is a hard
    constraint (via the tolerance band) and complexity is only minimized
    among near-equivalent-performance candidates. No weighted scalar
    fitness is introduced -- NSGA-II itself is untouched; this rule is
    applied only to already Pareto-optimal points, using validation data
    only. The official test set is never referenced by this function.
    """
    if not pareto:
        return {}, None

    highest_acc = max(pareto, key=lambda r: r["balanced_accuracy"])
    highest_composite = max(pareto, key=lambda r: r.get("validation_multi_metric_score", -1.0))
    lowest_complexity = min(pareto, key=lambda r: r["trainable_parameters"])

    # STEP 1
    max_val_bal_acc = highest_acc["balanced_accuracy"]
    # STEP 2 + STEP 3
    minimum_acceptable_balanced_accuracy = max_val_bal_acc - tolerance
    # STEP 4
    acceptable = [r for r in pareto
                  if r["balanced_accuracy"] >= minimum_acceptable_balanced_accuracy]
    # acceptable is guaranteed non-empty: highest_acc always satisfies its own criterion
    # STEP 5
    balanced = min(acceptable, key=lambda r: r["trainable_parameters"])

    selection_details = {
        "rule": "performance_constrained_min_params",
        "tolerance": tolerance,
        "max_val_balanced_accuracy": max_val_bal_acc,
        "minimum_acceptable_balanced_accuracy": minimum_acceptable_balanced_accuracy,
        "n_pareto_total": len(pareto),
        "n_acceptable_within_tolerance": len(acceptable),
        "acceptable_candidate_ids": [r.get("candidate_id") for r in acceptable],
        "selected_candidate_id": balanced.get("candidate_id"),
    }

    operating_points = {"highest_balanced_accuracy": highest_acc,
                         "highest_validation_multi_metric_score": highest_composite,
                         "lowest_complexity": lowest_complexity,
                         "balanced_tradeoff": balanced}
    return operating_points, selection_details


def describe_selection_rule(pareto: List[Dict[str, Any]],
                             operating_points: Dict[str, Dict[str, Any]],
                             selection_details: Optional[Dict[str, Any]]) -> str:
    """Produces a fully transparent, reproducible written description of
    the Pareto operating-point selection rule for architecture_summary.txt
    / experiment_log.txt.

    CURRENT RULE (performance-constrained, minimum-complexity):
      Among Pareto-optimal architectures, candidates within `tolerance`
      (default 0.01 = 1 percentage point) of the maximum validation
      Balanced Accuracy are considered acceptable; among those candidates,
      the architecture with the fewest trainable parameters is selected.

    This rule is applied ONLY to choose one practical point AFTER genuine
    NSGA-II Pareto optimization has already been performed. It is NOT a
    weighted single-objective fitness function, and it does NOT use the
    official test set at any point. The previously-used normalized
    Euclidean distance-to-ideal-point rule has been fully removed; no
    normalized accuracy/parameter values, ideal point, or distance
    calculation are used anywhere in this function."""
    lines = []
    lines.append("PARETO OPERATING-POINT SELECTION RULE (validation data only)")
    lines.append("=" * 70)
    lines.append("Objective 1 (MAXIMIZE): validation Balanced Accuracy")
    lines.append("Objective 2 (MINIMIZE): trainable parameter count")
    lines.append("NSGA-II is genuinely multi-objective (DEAP tools.selNSGA2, non-dominated")
    lines.append("sorting + crowding distance). The Pareto front is generated FIRST, from all")
    lines.append(f"valid evaluated candidates. Pareto-optimal candidates available: {len(pareto)}")
    lines.append("")
    if selection_details:
        tol = selection_details["tolerance"]
        lines.append("SELECTION RULE (performance-constrained, minimum-complexity):")
        lines.append("  STEP 1: Find the Pareto candidate with the maximum validation Balanced")
        lines.append(f"          Accuracy: maximum_validation_balanced_accuracy = "
                      f"{selection_details['max_val_balanced_accuracy']:.6f}")
        lines.append(f"  STEP 2: Define PERFORMANCE_TOLERANCE = {tol:.4f} "
                      f"({tol*100:.1f} percentage points of Balanced Accuracy).")
        lines.append("  STEP 3: minimum_acceptable_balanced_accuracy = "
                      "maximum_validation_balanced_accuracy - PERFORMANCE_TOLERANCE = "
                      f"{selection_details['minimum_acceptable_balanced_accuracy']:.6f}")
        lines.append("  STEP 4: Keep only Pareto candidates with "
                      "validation_balanced_accuracy >= minimum_acceptable_balanced_accuracy.")
        lines.append(f"          Acceptable candidates: {selection_details['n_acceptable_within_tolerance']} "
                      f"of {selection_details['n_pareto_total']} Pareto-optimal candidates "
                      f"(candidate_id(s): {selection_details['acceptable_candidate_ids']})")
        lines.append("  STEP 5: Among those acceptable candidates, select the one with the LOWEST")
        lines.append(f"          trainable parameter count -> selected candidate_id = "
                      f"{selection_details['selected_candidate_id']}")
        lines.append("")
        lines.append("Official test data (test accuracy, test Balanced Accuracy, test precision/")
        lines.append("recall/F1, test confusion matrix) is NOT used anywhere in this selection.")
        lines.append("This is NOT a weighted single-objective fitness function.")
        lines.append("This is NOT normalized-distance-to-ideal-point selection (that rule has")
        lines.append("been fully removed from this codebase).")
    else:
        lines.append("No Pareto-optimal candidates were available; a fallback rule was used")
        lines.append("(best-by-validation-Balanced-Accuracy among all valid candidates).")
    lines.append("")
    for key, label in [("highest_balanced_accuracy", "A) Highest Validation Balanced Accuracy"),
                        ("lowest_complexity", "B) Lowest Trainable-Parameter Count"),
                        ("balanced_tradeoff", "C) Performance-Constrained Minimum Complexity" )]:
        pt = operating_points.get(key)
        if pt:
            arch = json.loads(pt["architecture"])
            lines.append(f"[{label}]")
            lines.append(f"  candidate_id={pt.get('candidate_id')}  generation={pt.get('generation')}")
            lines.append(f"  val_balanced_accuracy={pt['balanced_accuracy']:.6f}  "
                          f"trainable_parameters={pt['trainable_parameters']}")
            lines.append(f"  architecture={arch}")
            lines.append("")
    return "\n".join(lines)


# ============================================================================
# 8. PLOTTING
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, title: str, out_path: Path) -> None:
    """sklearn confusion_matrix: rows = true labels, columns = predicted
    labels. Plotted accordingly: X-AXIS = Predicted, Y-AXIS = Actual."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # HS first row/col (true=rows, pred=cols)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["HS", "NHS"], yticklabels=["HS", "NHS"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ga_convergence(convergence_log: List[Dict[str, Any]], benchmark: str, out_path: Path) -> None:
    gens = [c["generation"] for c in convergence_log]
    vals = [c["best_balanced_accuracy"] for c in convergence_log]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gens, vals, marker="o", color="#2ca02c")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Validation Balanced Accuracy")
    ax.set_title(f"GA Convergence -- {BENCHMARK_DISPLAY[benchmark]}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pareto_front(all_records: List[Dict[str, Any]], pareto: List[Dict[str, Any]],
                       baseline_point: Dict[str, Any], operating_points: Dict[str, Dict[str, Any]],
                       benchmark: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    valid = [r for r in all_records if r.get("trainable_parameters") is not None
             and r.get("status") != "FAILED"]
    ax.scatter([r["trainable_parameters"] for r in valid],
               [r["balanced_accuracy"] for r in valid],
               c="lightgray", label="All valid GA candidates", s=35, zorder=2)

    if pareto:
        pareto_sorted = sorted(pareto, key=lambda r: r["trainable_parameters"])
        ax.plot([r["trainable_parameters"] for r in pareto_sorted],
                [r["balanced_accuracy"] for r in pareto_sorted],
                c="#2ca02c", marker="o", label="Pareto front", zorder=3)

    ax.scatter([baseline_point["trainable_parameters"]], [baseline_point["balanced_accuracy"]],
               c="#d62728", marker="*", s=250, label="Baseline (reference CNN)", zorder=4)

    markers = {"highest_balanced_accuracy": ("^", "#ff7f0e", "Highest BalAcc"),
               "lowest_complexity": ("s", "#9467bd", "Lowest complexity"),
               "balanced_tradeoff": ("D", "#17becf", "Balanced trade-off (selected)")}
    for key, (marker, color, label) in markers.items():
        pt = operating_points.get(key)
        if pt:
            ax.scatter([pt["trainable_parameters"]], [pt["balanced_accuracy"]],
                       c=color, marker=marker, s=140, edgecolor="black", label=label, zorder=5)

    ax.set_xlabel("Trainable Parameter Count")
    ax.set_ylabel("Validation Balanced Accuracy")
    ax.set_title(f"Pareto Front (from valid evaluated candidates) -- {BENCHMARK_DISPLAY[benchmark]}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_baseline_vs_ga_bar(values_baseline: List[float], values_ga: List[float],
                             labels: List[str], ylabel: str, title: str, out_path: Path) -> None:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, values_baseline, width, label="Baseline", color="#1f77b4")
    ax.bar(x + width / 2, values_ga, width, label="GA-Optimized", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ablation_comparison(ablation_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Plot the ablation study -- a CONTROL comparison, clearly separate
    from, and not a replacement for, the NSGA-II Pareto analysis."""
    df = pd.DataFrame(ablation_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    benchmarks = sorted(df["benchmark"].unique())
    ablations = df["ablation"].unique()
    x = np.arange(len(benchmarks))
    width = 0.8 / max(len(ablations), 1)
    for i, abl in enumerate(ablations):
        sub = df[df["ablation"] == abl].set_index("benchmark").reindex(benchmarks)
        ax.bar(x + i * width, sub["balanced_accuracy"], width, label=abl)
    ax.set_xticks(x + width * (len(ablations) - 1) / 2)
    ax.set_xticklabels([BENCHMARK_DISPLAY[b] for b in benchmarks])
    ax.set_ylabel("Test Balanced Accuracy")
    ax.set_title("Ablation Study (Control Comparison) -- NOT a replacement for NSGA-II")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================================
# 9. PER-BENCHMARK PIPELINE
# ============================================================================

def run_benchmark_pipeline(benchmark: str, cfg: Config, data_root: Path,
                            out_root: Path) -> Dict[str, Any]:
    LOGGER.info(f"===== Starting benchmark {BENCHMARK_DISPLAY[benchmark]} =====")
    bench_out = out_root / BENCHMARK_DISPLAY[benchmark]
    bench_out.mkdir(parents=True, exist_ok=True)
    corrupt_log: List[str] = []

    # ---- 1. discover + inspect + explicit resolution check ----
    paths = discover_dataset(data_root, benchmark)
    img_props = inspect_image_properties(paths)
    resolution_note = check_input_resolution(img_props, cfg, benchmark)
    LOGGER.info(f"[{benchmark}] Detected image properties: {img_props}")

    # ---- 2. class distribution analysis (dir-listing based, on FULL data) ----
    stats = analyze_dataset(paths, benchmark)
    pd.DataFrame([stats]).to_csv(bench_out / "dataset_statistics.csv", index=False)
    plot_class_distribution(stats, bench_out / "class_distribution.png")
    LOGGER.info(f"[{benchmark}] Class distribution: {stats}")

    # ---- 3. load actual images (official train/test respected) ----
    grayscale = img_props["most_common_mode"] not in ("RGB", "RGBA")
    cfg.grayscale = grayscale
    channels = 1 if grayscale else 3

    X_train_hs, y_train_hs = load_images(paths["train_hs"], 1, cfg.image_size, grayscale,
                                          cfg.max_images_per_class, corrupt_log, cfg.seed)
    X_train_nhs, y_train_nhs = load_images(paths["train_nhs"], 0, cfg.image_size, grayscale,
                                            cfg.max_images_per_class, corrupt_log, cfg.seed)
    X_test_hs, y_test_hs = load_images(paths["test_hs"], 1, cfg.image_size, grayscale,
                                        cfg.max_images_per_class, corrupt_log, cfg.seed)
    X_test_nhs, y_test_nhs = load_images(paths["test_nhs"], 0, cfg.image_size, grayscale,
                                          cfg.max_images_per_class, corrupt_log, cfg.seed)

    X_train_full = np.concatenate([X_train_hs, X_train_nhs], axis=0)
    y_train_full = np.concatenate([y_train_hs, y_train_nhs], axis=0)
    X_test = np.concatenate([X_test_hs, X_test_nhs], axis=0)
    y_test = np.concatenate([y_test_hs, y_test_nhs], axis=0)

    X_train_full = preprocess_images(X_train_full)
    X_test = preprocess_images(X_test)

    if corrupt_log:
        (bench_out / "corrupt_files_log.txt").write_text("\n".join(corrupt_log), encoding="utf-8")

    # ---- 4. official-train -> train/validation split (test set untouched) ----
    X_tr, y_tr, X_val, y_val = split_training_data(X_train_full, y_train_full, cfg.val_ratio, cfg.seed)
    LOGGER.info(
        f"[{benchmark}] LOADED SAMPLES: train={len(X_tr)} (HS={int(np.sum(y_tr == 1))}, NHS={int(np.sum(y_tr == 0))}); "
        f"validation={len(X_val)} (HS={int(np.sum(y_val == 1))}, NHS={int(np.sum(y_val == 0))}); "
        f"test={len(X_test)} (HS={int(np.sum(y_test == 1))}, NHS={int(np.sum(y_test == 0))}). "
        f"max_images_per_class={cfg.max_images_per_class}."
    )
    if cfg.max_images_per_class is not None:
        LOGGER.warning(
            f"[{benchmark}] SUBSET MODE: max_images_per_class={cfg.max_images_per_class} applies to BOTH official train and test class folders. "
            "Test metrics are exploratory subset metrics, not full official-test metrics."
        )

    input_shape = (cfg.image_size[0], cfg.image_size[1], channels)

    # ---- 5. baseline model ----
    LOGGER.info(f"[{benchmark}] Training baseline lightweight CNN ...")
    _clear_keras_session()
    baseline_model = build_baseline_model(input_shape, cfg.baseline_lr)
    baseline_summary_lines = []
    baseline_model.summary(print_fn=lambda l: baseline_summary_lines.append(l))
    baseline_summary_text = "\n".join(baseline_summary_lines)

    baseline_history = train_model(baseline_model, X_tr, y_tr, X_val, y_val, cfg.baseline_epochs,
                                    cfg.baseline_batch_size, cfg.early_stopping_patience, verbose=0,
                                    label=f"{benchmark}-baseline")
    baseline_trained = baseline_history is not None

    baseline_val_prob = baseline_model.predict(X_val, batch_size=cfg.baseline_batch_size, verbose=0).ravel()
    baseline_threshold, _ = choose_balanced_accuracy_threshold(y_val, baseline_val_prob)
    y_val_pred_b = (baseline_val_prob >= baseline_threshold).astype(int)
    baseline_test_prob = baseline_model.predict(X_test, batch_size=cfg.baseline_batch_size, verbose=0).ravel()
    y_test_pred_b = (baseline_test_prob >= baseline_threshold).astype(int)
    y_test_pred_b_05 = (baseline_test_prob >= 0.5).astype(int)

    baseline_val_metrics = calculate_metrics(y_val, y_val_pred_b)
    baseline_test_metrics = calculate_metrics(y_test, y_test_pred_b)  # validation-tuned threshold
    baseline_test_metrics_05 = calculate_metrics(y_test, y_test_pred_b_05)
    baseline_complexity = calculate_complexity(baseline_model)
    baseline_timing = measure_inference_time(baseline_model, X_test, cfg.inference_timing_repeats)

    baseline_row = {
        "benchmark": BENCHMARK_DISPLAY[benchmark], "model": "Baseline",
        "training_status": "ok" if baseline_trained else "SKIPPED_ZERO_EPOCHS_DEBUG",
        "decision_threshold": baseline_threshold,
        "test_threshold_policy": "validation_tuned_balanced_accuracy; frozen before test evaluation",
        **{f"val_{k}": v for k, v in baseline_val_metrics.items()},
        **{f"test_{k}": v for k, v in baseline_test_metrics.items()},
        **{f"test_fixed_0_5_{k}": v for k, v in baseline_test_metrics_05.items()},
        **baseline_complexity, **baseline_timing,
    }
    pd.DataFrame([baseline_row]).to_csv(bench_out / "baseline_metrics.csv", index=False)
    pd.DataFrame({"test_index": np.arange(len(y_test)), "y_true": y_test.astype(int),
                  "probability": baseline_test_prob.astype(float),
                  "validation_tuned_threshold": baseline_threshold, "y_pred_validation_tuned": y_test_pred_b.astype(int),
                  "fixed_threshold_0_5": 0.5, "y_pred_fixed_0_5": y_test_pred_b_05.astype(int)}).to_csv(
                      bench_out / "baseline_test_predictions.csv", index=False)
    plot_confusion_matrix(y_test, y_test_pred_b, f"Baseline Confusion Matrix -- {BENCHMARK_DISPLAY[benchmark]}",
                           bench_out / "baseline_confusion_matrix.png")
    LOGGER.info(f"[{benchmark}] Baseline TEST BalAcc={baseline_test_metrics['balanced_accuracy']:.4f} "
                f"params={baseline_complexity['trainable_params']} trained={baseline_trained}")

    # ---- 5b. independent hand-designed lightweight CNN comparator ----
    LOGGER.info(f"[{benchmark}] Training independent hand-designed separable CNN comparator ...")
    _clear_keras_session()
    hand_model = build_hand_designed_lightweight_model(input_shape, cfg.baseline_lr)
    hand_history = train_model(hand_model, X_tr, y_tr, X_val, y_val, cfg.baseline_epochs,
                               cfg.baseline_batch_size, cfg.early_stopping_patience, verbose=0,
                               label=f"{benchmark}-hand-designed")
    hand_val_prob = hand_model.predict(X_val, batch_size=cfg.baseline_batch_size, verbose=0).ravel()
    hand_threshold, _ = choose_balanced_accuracy_threshold(y_val, hand_val_prob)
    hand_test_prob = hand_model.predict(X_test, batch_size=cfg.baseline_batch_size, verbose=0).ravel()
    hand_test_pred = (hand_test_prob >= hand_threshold).astype(int)
    hand_test_pred_05 = (hand_test_prob >= 0.5).astype(int)
    hand_val_metrics = calculate_metrics(y_val, (hand_val_prob >= hand_threshold).astype(int))
    hand_test_metrics = calculate_metrics(y_test, hand_test_pred)
    hand_test_metrics_05 = calculate_metrics(y_test, hand_test_pred_05)
    hand_complexity = calculate_complexity(hand_model)
    hand_timing = measure_inference_time(hand_model, X_test, cfg.inference_timing_repeats)
    hand_row = {
        "benchmark": BENCHMARK_DISPLAY[benchmark], "model": "Hand-designed lightweight CNN",
        "training_status": "ok" if hand_history is not None else "SKIPPED_ZERO_EPOCHS_DEBUG",
        "decision_threshold": hand_threshold,
        "test_threshold_policy": "validation_tuned_balanced_accuracy; frozen before test evaluation",
        **{f"val_{k}": v for k, v in hand_val_metrics.items()},
        **{f"test_{k}": v for k, v in hand_test_metrics.items()},
        **{f"test_fixed_0_5_{k}": v for k, v in hand_test_metrics_05.items()},
        **hand_complexity, **hand_timing,
    }
    pd.DataFrame([hand_row]).to_csv(bench_out / "hand_designed_metrics.csv", index=False)
    pd.DataFrame({"test_index": np.arange(len(y_test)), "y_true": y_test.astype(int),
                  "probability": hand_test_prob.astype(float), "validation_tuned_threshold": hand_threshold,
                  "y_pred_validation_tuned": hand_test_pred.astype(int)}).to_csv(
                      bench_out / "hand_designed_test_predictions.csv", index=False)
    plot_confusion_matrix(y_test, hand_test_pred,
                          f"Hand-designed Lightweight CNN -- {BENCHMARK_DISPLAY[benchmark]}",
                          bench_out / "hand_designed_confusion_matrix.png")
    LOGGER.info(f"[{benchmark}] Hand-designed CNN TEST BalAcc={hand_test_metrics['balanced_accuracy']:.4f} "
                f"params={hand_complexity['trainable_params']}")
    del hand_model
    _clear_keras_session()

    # ---- 6. multi-objective GA (genuine NSGA-II) ----
    LOGGER.info(f"[{benchmark}] Running NSGA-II GA search "
                f"(pop={cfg.population_size}, gens={cfg.generations}, search_epochs={cfg.search_epochs}) ...")
    ga_records, convergence_log = run_nsga2(input_shape, X_tr, y_tr, X_val, y_val, cfg, benchmark)
    ga_df = pd.DataFrame(ga_records)
    ga_df.to_csv(bench_out / "ga_candidates.csv", index=False)
    plot_ga_convergence(convergence_log, benchmark, bench_out / "ga_convergence.png")

    # ---- 7. Pareto front (validation-only selection, never test) ----
    pareto = identify_pareto_front(ga_records)
    pareto_df = pd.DataFrame(pareto)
    pareto_df.to_csv(bench_out / "pareto_front.csv", index=False)
    operating_points, selection_details = select_operating_points(pareto)

    # GUARANTEE: every benchmark must yield a baseline row AND a GA row, even
    # if the Pareto front is unexpectedly empty (e.g. every candidate failed).
    if not operating_points:
        valid_candidates = [r for r in ga_records
                             if r.get("status") != "FAILED" and r.get("trainable_parameters") is not None]
        if not valid_candidates:
            raise RuntimeError(
                f"[{benchmark}] Every GA candidate failed to train -- cannot produce a GA model "
                f"for this benchmark. Check master_experiment_log.txt for the underlying errors."
            )
        LOGGER.warning(f"[{benchmark}] Pareto front was empty; falling back to the best-by-validation-"
                        f"BalAcc GA candidate so a GA model is still produced for this benchmark.")
        fallback = max(valid_candidates, key=lambda r: r["balanced_accuracy"])
        operating_points = {"highest_balanced_accuracy": fallback,
                             "highest_validation_multi_metric_score": fallback,
                             "lowest_complexity": fallback,
                             "balanced_tradeoff": fallback}
        selection_details = None

    pd.DataFrame(list(operating_points.values())).to_csv(bench_out / "selected_operating_points.csv", index=False)
    selection_rule_text = describe_selection_rule(pareto, operating_points, selection_details)

    baseline_point_for_plot = {
        "trainable_parameters": baseline_complexity["trainable_params"],
        "balanced_accuracy": baseline_val_metrics["balanced_accuracy"],
    }
    plot_pareto_front(ga_records, pareto, baseline_point_for_plot, operating_points,
                       benchmark, bench_out / "pareto_front.png")
    LOGGER.info(f"[{benchmark}] Pareto front size = {len(pareto)} / {len(ga_records)} valid candidates")
    hypervolume_report = calculate_pareto_hypervolume(ga_records)
    (bench_out / "pareto_hypervolume.json").write_text(json.dumps(hypervolume_report, indent=2), encoding="utf-8")
    LOGGER.info(f"[{benchmark}] Normalized Pareto hypervolume = {hypervolume_report['hypervolume']:.6f} (reference={hypervolume_report['reference_point']})")

    # ---- 8. final GA model: prioritize the Pareto candidate with highest validation balanced accuracy ----
    # Protocol: trained on the 80% official-train split ONLY; the validation
    # split is used purely for early-stopping monitoring and is NOT merged
    # into the training data, mirroring the baseline procedure.
    chosen = operating_points["highest_balanced_accuracy"]
    chosen_chromosome = Chromosome(**json.loads(chosen["architecture"]))
    LOGGER.info(f"[{benchmark}] Retraining selected GA architecture: {chosen_chromosome}")

    _clear_keras_session()
    final_model = build_candidate_model(chosen_chromosome, input_shape)
    final_history = train_model(final_model, X_tr, y_tr, X_val, y_val, cfg.final_epochs,
                                 cfg.ga_batch_size, cfg.early_stopping_patience, verbose=0,
                                 label=f"{benchmark}-final-GA")
    final_trained = final_history is not None

    final_val_prob = final_model.predict(X_val, batch_size=cfg.ga_batch_size, verbose=0).ravel()
    final_threshold, _ = choose_balanced_accuracy_threshold(y_val, final_val_prob)
    y_val_pred_ga = (final_val_prob >= final_threshold).astype(int)
    final_test_prob = final_model.predict(X_test, batch_size=cfg.ga_batch_size, verbose=0).ravel()
    y_test_pred_ga = (final_test_prob >= final_threshold).astype(int)
    y_test_pred_ga_05 = (final_test_prob >= 0.5).astype(int)
    final_val_metrics = calculate_metrics(y_val, y_val_pred_ga)
    final_test_metrics = calculate_metrics(y_test, y_test_pred_ga)  # validation-tuned threshold
    final_test_metrics_05 = calculate_metrics(y_test, y_test_pred_ga_05)
    final_complexity = calculate_complexity(final_model)
    final_timing = measure_inference_time(final_model, X_test, cfg.inference_timing_repeats)

    model_label = "GA" if final_trained else "GA [DEBUG SMOKE TEST - NOT TRAINED, epochs=0]"
    final_metrics_row = {
        "benchmark": BENCHMARK_DISPLAY[benchmark], "model": model_label,
        "training_status": "ok" if final_trained else "SKIPPED_ZERO_EPOCHS_DEBUG",
        "selected_operating_point": "highest_validation_balanced_accuracy",
        "decision_threshold": final_threshold,
        "test_threshold_policy": "validation_tuned_balanced_accuracy; frozen before test evaluation",
        "search_fitness_threshold": "validation-tuned balanced-accuracy threshold",
        "architecture": json.dumps(asdict(chosen_chromosome)),
        **{f"val_{k}": v for k, v in final_val_metrics.items()},
        **{f"test_{k}": v for k, v in final_test_metrics.items()},
        **{f"test_fixed_0_5_{k}": v for k, v in final_test_metrics_05.items()},
        **final_complexity, **final_timing,
    }
    pd.DataFrame([final_metrics_row]).to_csv(bench_out / "final_metrics.csv", index=False)
    pd.DataFrame({"test_index": np.arange(len(y_test)), "y_true": y_test.astype(int),
                  "probability": final_test_prob.astype(float),
                  "validation_tuned_threshold": final_threshold, "y_pred_validation_tuned": y_test_pred_ga.astype(int),
                  "fixed_threshold_0_5": 0.5, "y_pred_fixed_0_5": y_test_pred_ga_05.astype(int)}).to_csv(
                      bench_out / "ga_test_predictions.csv", index=False)
    plot_confusion_matrix(y_test, y_test_pred_ga,
                           f"GA-Optimized Confusion Matrix -- {BENCHMARK_DISPLAY[benchmark]}",
                           bench_out / "ga_confusion_matrix.png")
    LOGGER.info(f"[{benchmark}] GA-optimized TEST BalAcc={final_test_metrics['balanced_accuracy']:.4f} "
                f"params={final_complexity['trainable_params']} trained={final_trained}")
    del final_model
    _clear_keras_session()

    # ---- 9. architecture_summary.txt (baseline summary + candidate family + selection rule) ----
    complexity_summary_text = (
        "COMPLEXITY REPORTING NOTE\n"
        "========================\n"
        "Trainable parameter count is the primary complexity metric. The following\n"
        "model-specific parameter and operation counts are calculated from the\n"
        "instantiated Keras models; no fixed FLOPs/MACs value is hardcoded.\n\n"
        + describe_complexity_report(baseline_complexity, "Baseline CNN") + "\n\n"
        + describe_complexity_report(final_complexity, "Selected GA CNN")
    )
    architecture_summary_text = (
        baseline_summary_text + "\n\n" +
        CANDIDATE_ARCHITECTURE_DEFINITION + "\n\n" + GA_OPERATOR_DEFINITION + "\n\n" +
        selection_rule_text + "\n\n" + complexity_summary_text
    )
    (bench_out / "architecture_summary.txt").write_text(architecture_summary_text, encoding="utf-8")

    # ---- 10. ablation study (CONTROL comparison, does NOT replace NSGA-II) ----
    # Ablation 1: reference lightweight CNN (baseline, no GA at all)
    # Ablation 2: highest-validation-Balanced-Accuracy candidate from the same NSGA-II pool.
    # This is NOT an independently optimized single-objective GA.
    # Ablation 3: multi-objective NSGA-II GA (the main method, from step 8 above)
    LOGGER.info(f"[{benchmark}] Running ablation: highest-validation-Balanced-Accuracy candidate from the NSGA-II pool ...")
    ablation_rows = [
        {"benchmark": BENCHMARK_DISPLAY[benchmark],
         "ablation": "Ablation 1: Reference lightweight CNN (baseline, no GA)",
         "balanced_accuracy": baseline_test_metrics["balanced_accuracy"],
         "precision": baseline_test_metrics["precision"], "recall": baseline_test_metrics["recall"],
         "specificity": baseline_test_metrics["specificity"], "f1": baseline_test_metrics["f1"],
         "decision_threshold": baseline_threshold,
         **{f"test_fixed_0_5_{k}": v for k, v in baseline_test_metrics_05.items()},
         "trainable_parameters": baseline_complexity["trainable_params"]},
        {"benchmark": BENCHMARK_DISPLAY[benchmark],
         "ablation": "Ablation 3: Multi-objective NSGA-II GA (BalAcc + params)",
         "balanced_accuracy": final_test_metrics["balanced_accuracy"],
         "precision": final_test_metrics["precision"], "recall": final_test_metrics["recall"],
         "specificity": final_test_metrics["specificity"], "f1": final_test_metrics["f1"],
         "decision_threshold": final_threshold,
         **{f"test_fixed_0_5_{k}": v for k, v in final_test_metrics_05.items()},
         "trainable_parameters": final_complexity["trainable_params"]},
    ]

    # Bug fix: previously a candidate with a FAILED status but a leftover
    # non-None trainable_parameters value (or vice versa) could slip through
    # and break max()/downstream comparisons. The filter below is explicit
    # and matches the required logic exactly.
    valid_acc_candidates = [
        r for r in ga_records
        if not r.get("FAILED", False) and r.get("status") != "FAILED"
        and r.get("trainable_parameters") is not None
    ]
    if not valid_acc_candidates:
        raise RuntimeError(f"[{benchmark}] No valid GA candidates available for accuracy-only ablation.")

    acc_only_candidate = max(valid_acc_candidates, key=lambda r: r["balanced_accuracy"])  # validation-only selection
    acc_only_chromosome = Chromosome(**json.loads(acc_only_candidate["architecture"]))
    _clear_keras_session()
    acc_model = build_candidate_model(acc_only_chromosome, input_shape)
    train_model(acc_model, X_tr, y_tr, X_val, y_val, cfg.final_epochs, cfg.ga_batch_size,
                cfg.early_stopping_patience, verbose=0, label=f"{benchmark}-ablation-acc-only")
    acc_val_prob = acc_model.predict(X_val, batch_size=cfg.ga_batch_size, verbose=0).ravel()
    acc_threshold, _ = choose_balanced_accuracy_threshold(y_val, acc_val_prob)
    acc_test_prob = acc_model.predict(X_test, batch_size=cfg.ga_batch_size, verbose=0).ravel()
    y_test_pred_acc = (acc_test_prob >= acc_threshold).astype(int)
    y_test_pred_acc_05 = (acc_test_prob >= 0.5).astype(int)
    acc_only_test_metrics = calculate_metrics(y_test, y_test_pred_acc)
    acc_only_test_metrics_05 = calculate_metrics(y_test, y_test_pred_acc_05)
    pd.DataFrame({"test_index": np.arange(len(y_test)), "y_true": y_test.astype(int),
                  "probability": acc_test_prob.astype(float),
                  "validation_tuned_threshold": acc_threshold, "y_pred_validation_tuned": y_test_pred_acc.astype(int),
                  "fixed_threshold_0_5": 0.5, "y_pred_fixed_0_5": y_test_pred_acc_05.astype(int)}).to_csv(
                      bench_out / "accuracy_only_ga_test_predictions.csv", index=False)
    acc_only_complexity = calculate_complexity(acc_model)
    ablation_rows.append({
        "benchmark": BENCHMARK_DISPLAY[benchmark],
        "ablation": "Ablation 2: Highest-validation-Balanced-Accuracy candidate from NSGA-II pool (not independent accuracy-only GA)",
        "decision_threshold": acc_threshold,
        "balanced_accuracy": acc_only_test_metrics["balanced_accuracy"],
        "precision": acc_only_test_metrics["precision"], "recall": acc_only_test_metrics["recall"],
        "specificity": acc_only_test_metrics["specificity"], "f1": acc_only_test_metrics["f1"],
        **{f"test_fixed_0_5_{k}": v for k, v in acc_only_test_metrics_05.items()},
        "trainable_parameters": acc_only_complexity["trainable_params"],
    })
    del acc_model
    _clear_keras_session()

    pd.DataFrame(ablation_rows).to_csv(bench_out / "ablation_results.csv", index=False)

    # ---- 11. summary plots for this benchmark ----
    labels = [BENCHMARK_DISPLAY[benchmark]]
    plot_baseline_vs_ga_bar([baseline_test_metrics["balanced_accuracy"]],
                             [final_test_metrics["balanced_accuracy"]], labels,
                             "Test Balanced Accuracy", f"Baseline vs GA -- {BENCHMARK_DISPLAY[benchmark]}",
                             bench_out / "baseline_vs_ga_balanced_accuracy.png")
    plot_baseline_vs_ga_bar([baseline_complexity["trainable_params"]],
                             [final_complexity["trainable_params"]], labels,
                             "Trainable Parameters", f"Baseline vs GA Params -- {BENCHMARK_DISPLAY[benchmark]}",
                             bench_out / "baseline_vs_ga_params.png")
    plot_baseline_vs_ga_bar([baseline_timing["inference_time_ms_mean"]],
                             [final_timing["inference_time_ms_mean"]], labels,
                             "Inference Time (ms/image)", f"Baseline vs GA Inference Time -- {BENCHMARK_DISPLAY[benchmark]}",
                             bench_out / "baseline_vs_ga_inference_time.png")

    # ---- 12. experiment_log.txt ----
    exp_log = {
        "benchmark": benchmark, "image_properties": img_props,
        "resolution_check": resolution_note,
        "pixel_preprocessing_note": (
            f"All loaded images are resized to {cfg.image_size[0]}x{cfg.image_size[1]} pixels "
            "and normalized to float32 [0,1]. This is the configured preprocessing resolution, "
            "not a claim that the source images were natively this size."
        ),
        "complexity_reporting_note": {
            "primary_metric": "trainable parameter count",
            "baseline": describe_complexity_report(baseline_complexity, "Baseline CNN"),
            "selected_ga": describe_complexity_report(final_complexity, "Selected GA CNN"),
        },
        "dataset_statistics": stats, "n_train": len(X_tr), "n_val": len(X_val),
        "n_test": len(X_test), "corrupt_files": len(corrupt_log),
        "max_images_per_class": cfg.max_images_per_class,
        "test_evaluation_scope": ("SUBSET_ONLY_EXPLORATORY" if cfg.max_images_per_class is not None else "all_loadable_images_in_official_test_folders"),
        "threshold_reporting": "test_* metrics use the validation-tuned threshold; test_fixed_0_5_* metrics use threshold 0.5.",
        "ga_search_threshold": "validation-tuned threshold per candidate; selected without test labels",
        "final_training_protocol": (
            "Final GA model retrained using the 80% official-train split ONLY. "
            "The 20% validation split was used purely for early-stopping monitoring "
            "and was NOT merged into the training data (mirrors baseline procedure). "
            "Within this run, the official test set was held out from training, validation, and architecture selection. "
            "The developer has examined test results from earlier exploratory runs, so project-wide test-set blindness "
            "is not claimed. Test metrics never drive selection within this run."
        ),
        "pareto_selection_rule": selection_rule_text,
        "baseline": baseline_row, "final_ga": final_metrics_row,
        "n_ga_candidates_evaluated": len(ga_records), "n_pareto_optimal": len(pareto),
    }
    (bench_out / "experiment_log.txt").write_text(json.dumps(exp_log, indent=2, default=str), encoding="utf-8")

    return {
    "benchmark": benchmark,
    "baseline_row": baseline_row,
    "hand_row": hand_row,
    "final_metrics_row": final_metrics_row,
    "ablation_rows": ablation_rows,
    "resolution_note": resolution_note,
    "n_test": len(X_test),
}

# ============================================================================
# 10. OVERALL AGGREGATION / EXCEL REPORT
# ============================================================================

FULL_METRIC_COLUMNS = [
    "Benchmark", "Model",
    "Accuracy", "Balanced Accuracy", "Precision", "Recall/Sensitivity", "Specificity", "F1-score",
    "Fixed-0.5 Accuracy", "Fixed-0.5 Balanced Accuracy", "Fixed-0.5 Precision",
    "Fixed-0.5 Recall/Sensitivity", "Fixed-0.5 Specificity", "Fixed-0.5 F1-score",
    "TP", "TN", "FP", "FN",
    "Parameters", "Trainable Parameters", "Estimated MACs (Conv/Dense only)",
    "Estimated FLOPs (2x MACs)", "Complexity Measurement Note", "Inference Time (ms)",
]


def _row_from_result(row: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Builds one FULL_METRIC_COLUMNS row from a baseline_row / final_metrics_row
    dict. Nothing is dropped or hidden in a separate/trimmed dataframe."""
    return {
        "Benchmark": row["benchmark"], "Model": model_name,
        "Accuracy": row["test_accuracy"],
        "Balanced Accuracy": row["test_balanced_accuracy"],
        "Precision": row["test_precision"],
        "Recall/Sensitivity": row["test_recall"],
        "Specificity": row["test_specificity"],
        "F1-score": row["test_f1"],
        "Fixed-0.5 Accuracy": row["test_fixed_0_5_accuracy"],
        "Fixed-0.5 Balanced Accuracy": row["test_fixed_0_5_balanced_accuracy"],
        "Fixed-0.5 Precision": row["test_fixed_0_5_precision"],
        "Fixed-0.5 Recall/Sensitivity": row["test_fixed_0_5_recall"],
        "Fixed-0.5 Specificity": row["test_fixed_0_5_specificity"],
        "Fixed-0.5 F1-score": row["test_fixed_0_5_f1"],
        "TP": row["test_TP"], "TN": row["test_TN"], "FP": row["test_FP"], "FN": row["test_FN"],
        "Parameters": row["total_params"],
        "Trainable Parameters": row["trainable_params"],
        "Estimated MACs (Conv/Dense only)": row["estimated_macs"],
        "Estimated FLOPs (2x MACs)": row["estimated_flops"],
        "Complexity Measurement Note": row.get("flops_measurement_note", "Not available."),
        "Inference Time (ms)": row["inference_time_ms_mean"],
    }


def build_overall_comparison(all_results: List[Dict[str, Any]], out_root: Path) -> pd.DataFrame:
    rows = []
    for res in all_results:
        rows.append(_row_from_result(res["baseline_row"], "Baseline"))
        rows.append(_row_from_result(res["hand_row"], "Hand-designed lightweight CNN"))
        rows.append(_row_from_result(res["final_metrics_row"], "GA"))
    df = pd.DataFrame(rows, columns=FULL_METRIC_COLUMNS)
    df.to_csv(out_root / "overall" / "baseline_vs_ga.csv", index=False)
    return df


# Raw TP/TN/FP/FN counts are not averaged. Operation counts are partial
# Conv2D/Dense estimates and are averaged only as supplementary indicators.
MEAN_AGGREGABLE_COLUMNS = [
    "Accuracy", "Balanced Accuracy", "Precision", "Recall/Sensitivity", "Specificity",
    "F1-score", "Fixed-0.5 Accuracy", "Fixed-0.5 Balanced Accuracy",
    "Fixed-0.5 Precision", "Fixed-0.5 Recall/Sensitivity", "Fixed-0.5 Specificity",
    "Fixed-0.5 F1-score", "Parameters", "Trainable Parameters", "Inference Time (ms)",
]


def build_mean_summary(comparison_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Unweighted benchmark-level means, including supplementary MAC/FLOP estimates."""
    df = comparison_df.copy()
    df["_model_group"] = df["Model"]

    mean_columns = MEAN_AGGREGABLE_COLUMNS + [
        "Estimated MACs (Conv/Dense only)",
        "Estimated FLOPs (2x MACs)",
    ]
    for col in mean_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_mean_columns = [col for col in mean_columns if col in df.columns]
    mean_df = df.groupby("_model_group", as_index=False)[available_mean_columns].mean(numeric_only=True)
    mean_df = mean_df.rename(columns={"_model_group": "Model"})

    # Keep the output schema stable if an optional supplementary column is absent.
    for col in FULL_METRIC_COLUMNS:
        if col not in mean_df.columns and col not in ("Benchmark", "Model", "Complexity Measurement Note"):
            mean_df[col] = np.nan

    # Confusion counts cannot be meaningfully averaged across separate test sets.
    for col in ["TP", "TN", "FP", "FN"]:
        mean_df[col] = None

    mean_df["Complexity Measurement Note"] = (
        "Unweighted mean of benchmark-level estimates; Conv2D/Dense MACs only; "
        "FLOPs = 2 x MACs; excludes BatchNorm, pooling, activations, bias adds and memory movement."
    )
    mean_df.insert(0, "Benchmark", "MEAN (benchmark-level average across ICCAD1-5)")
    mean_df = mean_df[FULL_METRIC_COLUMNS]
    mean_df.to_csv(out_root / "overall" / "overall_summary.csv", index=False)
    return mean_df


def create_excel_report(comparison_df: pd.DataFrame, mean_df: pd.DataFrame,
                         all_dataset_stats: pd.DataFrame, all_ga_candidates: pd.DataFrame,
                         all_pareto: pd.DataFrame, all_ablation: pd.DataFrame,
                         out_root: Path) -> None:
    """Writes G13_results.xlsx with all 7 required sheets, each carrying the
    full metric set -- nothing hidden in a trimmed/separate dataframe."""
    xlsx_path = out_root / "overall" / "G13_results.xlsx"
    overall = pd.concat([comparison_df, mean_df], ignore_index=True)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        all_dataset_stats.to_excel(writer, sheet_name="Dataset Statistics", index=False)
        comparison_df[comparison_df["Model"] == "Baseline"].to_excel(
            writer, sheet_name="Baseline Results", index=False)
        all_ga_candidates.to_excel(writer, sheet_name="GA Candidate Results", index=False)
        all_pareto.to_excel(writer, sheet_name="Pareto Results", index=False)
        overall.to_excel(writer, sheet_name="Final Results", index=False)
        all_ablation.to_excel(writer, sheet_name="Ablation Results", index=False)
        overall.to_excel(writer, sheet_name="Overall Comparison", index=False)
    LOGGER.info(f"Excel report written to {xlsx_path}")


def plot_overall_comparisons(comparison_df: pd.DataFrame, out_root: Path) -> None:
    benchmarks = comparison_df["Benchmark"].unique()
    base_bal = [comparison_df[(comparison_df.Benchmark == b) & (comparison_df.Model == "Baseline")]["Balanced Accuracy"].values[0]
                for b in benchmarks]
    hand_bal_vals, ga_bal_vals = [], []
    for b in benchmarks:
        hand = comparison_df[(comparison_df.Benchmark == b) & (comparison_df.Model == "Hand-designed lightweight CNN")]["Balanced Accuracy"]
        ga = comparison_df[(comparison_df.Benchmark == b) & (comparison_df.Model == "GA")]["Balanced Accuracy"]
        hand_bal_vals.append(hand.values[0] if len(hand) else np.nan)
        ga_bal_vals.append(ga.values[0] if len(ga) else np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(benchmarks))
    width = 0.25
    ax.bar(x - width, base_bal, width, label="Baseline", color="#1f77b4")
    ax.bar(x, hand_bal_vals, width, label="Hand-designed lightweight CNN", color="#ff7f0e")
    ax.bar(x + width, ga_bal_vals, width, label="GA-Optimized", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Test Balanced Accuracy")
    ax.set_title("Overall Benchmark Comparison -- Baseline vs GA")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_root / "overall" / "overall_benchmark_comparison.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 11. MAIN
# ============================================================================

def main() -> None:
    cfg = build_config_from_args()
    data_root = Path(cfg.data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {data_root}")
    missing_benchmarks = [b for b in BENCHMARKS if not (data_root / b).is_dir()]
    if missing_benchmarks:
        raise FileNotFoundError(
            f"Dataset root is missing benchmark folders: {missing_benchmarks}. "
            f"Expected iccad1 through iccad5 under {data_root}"
        )
    base_output_dir = Path(cfg.output_dir)
    out_root = resolve_output_dir(base_output_dir, cfg.mode, cfg.overwrite)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "overall").mkdir(parents=True, exist_ok=True)

    setup_logger(out_root / "master_experiment_log.txt")
    set_global_seed(cfg.seed)

    if out_root != base_output_dir:
        LOGGER.warning(f"Output directory redirected to avoid overwriting existing results: {out_root}")

    LOGGER.info(f"Python {sys.version} | TensorFlow {tf.__version__} | Platform {platform.platform()}")
    LOGGER.info(f"Mode = {cfg.mode.upper()} | Config = {asdict(cfg)}")

    config_dump = {
        **asdict(cfg),
        "resolved_output_dir": str(out_root),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(data_root),
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "candidate_gene_space": {**GENE_SPACE, "dropout_range": DROPOUT_RANGE, "learning_rate_range": LR_RANGE},
        "ga_operators_documentation": GA_OPERATOR_DEFINITION,
        "full_mode_default_search_budget": {"population": 24, "generations": 12, "search_epochs": 8},
        "comparison_models": ["reference baseline CNN", "independent hand-designed separable lightweight CNN", "NSGA-II GA CNN"],
        "repeated_runs_protocol": "Repeat full experiments with distinct --seed values (e.g. 11, 22, 33, 44, 55); aggregate mean and standard deviation across completed runs. This script records seed in experiment_config.json; do not claim repeated-run statistics from a single seed.",
        "hypervolume_method": "Per-benchmark normalized 2-D hypervolume; maximize (balanced accuracy, normalized parameter efficiency), reference point (0,0); see each benchmark pareto_hypervolume.json.",
        "ga_objectives": ["MAXIMIZE validation Balanced Accuracy", "MINIMIZE trainable parameter count"],
        "performance_claim_guardrail": "The code cannot guarantee GA test balanced accuracy exceeds baseline. Report actual held-out test outcomes honestly; do not select models based on test results.",
        "final_pareto_candidate_selection": "Select the Pareto candidate with the highest validation Balanced Accuracy for final retraining; also report lower-complexity trade-off points separately.",
        "ga_fitness_threshold": "Each GA candidate threshold is selected on validation data to maximize validation Balanced Accuracy; fitness is that validation Balanced Accuracy. The official test set is not used for selection.",
        "threshold_selection": "For final baseline, selected GA model, and ablation model, select threshold by maximizing validation Balanced Accuracy only; freeze it before test evaluation. Test metrics are also reported at fixed threshold 0.5.",
        "test_set_caveat": "Test labels are not used for selection within a run. The developer has examined results from earlier pilot runs, so project-wide test-set blindness cannot be claimed; final claims should ideally use an untouched holdout.",
        "subset_policy": "When max_images_per_class is set, deterministic random sampling is applied to every class folder, including test. Such test results are explicitly subset-only and exploratory.",
        "pareto_selection_rule_summary": (
            "NSGA-II maximizes validation Balanced Accuracy and minimizes trainable parameters. "
            "The final reported GA model is retrained from the Pareto candidate with the highest "
            "validation Balanced Accuracy. Lowest-complexity and performance-constrained tradeoff "
            "points are also retained in selected_operating_points.csv for comparison. "
            "The classification threshold is selected on validation probabilities only and then "
            "frozen for the one-time official-test evaluation. No test data are used for selection."
        ),
        "final_training_protocol": (
            "Final GA model and baseline are both trained on the 80% official-train split "
            "ONLY; the 20% validation split is used purely for early-stopping monitoring and "
            "is never merged into training data. The official test set is used exclusively "
            "for one-time final evaluation, after architecture selection is complete."
        ),
    }
    (out_root / "experiment_config.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    all_results = []
    all_dataset_stats_rows = []
    all_ga_candidates_frames = []
    all_pareto_frames = []
    all_ablation_frames = []
    benchmark_status_rows = []

    for benchmark in BENCHMARKS:
        try:
            result = run_benchmark_pipeline(benchmark, cfg, data_root, out_root)
            all_results.append(result)
            bstats_path = out_root / BENCHMARK_DISPLAY[benchmark] / "dataset_statistics.csv"
            all_dataset_stats_rows.append(pd.read_csv(bstats_path))
            all_ga_candidates_frames.append(pd.read_csv(out_root / BENCHMARK_DISPLAY[benchmark] / "ga_candidates.csv"))
            all_pareto_frames.append(pd.read_csv(out_root / BENCHMARK_DISPLAY[benchmark] / "pareto_front.csv"))
            all_ablation_frames.append(pd.DataFrame(result["ablation_rows"]))
            benchmark_status_rows.append({"benchmark": BENCHMARK_DISPLAY[benchmark], "status": "COMPLETED", "error": None})
        except Exception as e:
            LOGGER.error(f"Benchmark {benchmark} FAILED entirely: {e}\n{traceback.format_exc()}")
            benchmark_status_rows.append({"benchmark": BENCHMARK_DISPLAY[benchmark], "status": "FAILED", "error": str(e)})
            continue

    pd.DataFrame(benchmark_status_rows).to_csv(out_root / "overall" / "benchmark_status.csv", index=False)
    failed_benchmarks = [r["benchmark"] for r in benchmark_status_rows if r["status"] != "COMPLETED"]
    if not all_results:
        (out_root / "overall" / "RUN_STATUS.txt").write_text(
            "FAILED RUN\nNo benchmark completed successfully. See master_experiment_log.txt.\n",
            encoding="utf-8"
        )
        LOGGER.error("No benchmark completed successfully. Aborting overall aggregation.")
        return

    comparison_df = build_overall_comparison(all_results, out_root)
    mean_df = build_mean_summary(comparison_df, out_root)
    all_dataset_stats = pd.concat(all_dataset_stats_rows, ignore_index=True) if all_dataset_stats_rows else pd.DataFrame()
    all_ga_candidates = pd.concat(all_ga_candidates_frames, ignore_index=True) if all_ga_candidates_frames else pd.DataFrame()
    all_pareto = pd.concat(all_pareto_frames, ignore_index=True) if all_pareto_frames else pd.DataFrame()
    all_ablation = pd.concat(all_ablation_frames, ignore_index=True) if all_ablation_frames else pd.DataFrame()

    create_excel_report(comparison_df, mean_df, all_dataset_stats, all_ga_candidates,
                         all_pareto, all_ablation, out_root)
    plot_overall_comparisons(comparison_df, out_root)

    if failed_benchmarks:
        (out_root / "overall" / "RUN_STATUS.txt").write_text(
            "INCOMPLETE RUN\nFailed or missing benchmark(s): " + ", ".join(failed_benchmarks) +
            "\nOverall averages below cover only successfully completed benchmarks.\n",
            encoding="utf-8"
        )
        LOGGER.error("===== EXPERIMENT INCOMPLETE: failed benchmark(s): %s =====", ", ".join(failed_benchmarks))
    else:
        (out_root / "overall" / "RUN_STATUS.txt").write_text(
            "ALL FIVE BENCHMARKS COMPLETED SUCCESSFULLY\n", encoding="utf-8"
        )
        LOGGER.info("===== EXPERIMENT COMPLETE: all five benchmarks succeeded =====")
    LOGGER.info(f"Results written under: {out_root.resolve()}")
    LOGGER.info(f"\n{mean_df.to_string(index=False)}")


if __name__ == "__main__":
    # Visible marker: confirms this exact file actually entered the experiment.
    print("[G13] Entry point reached; starting experiment...", flush=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[G13] Interrupted by user (Ctrl+C).", flush=True)
        raise
    except Exception as exc:
        print(f"[G13] FATAL ERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
