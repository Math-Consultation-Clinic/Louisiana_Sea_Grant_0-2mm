



from __future__ import print_function, unicode_literals, absolute_import, division

import os
import csv
import math
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy import ndimage as ndi

from csbdeep.utils import normalize
from stardist import fill_label_holes, calculate_extents
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_IMAGE_DIR = "/scratch/gsunka1/LSG_0-2mm/original"
DEFAULT_MASK_DIR  = "/scratch/gsunka1/LSG_0-2mm/cellpose_outputs/masks_tif"
DEFAULT_OUTPUT_DIR = "/scratch/gsunka1/LSG_0-2mm/stardist_results"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Working StarDist training script for oyster 0-2mm segmentation."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Folder containing original oyster images."
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        default=DEFAULT_MASK_DIR,
        help="Folder containing annotation masks."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where models/results will be saved."
    )

    parser.add_argument(
        "--dataset_size",
        type=int,
        default=None,
        help="Number of images used for train+validation. Default: all except test images."
    )

    parser.add_argument(
        "--testing_size",
        type=int,
        default=1,
        help="Number of test images. Default: 10."
    )

    parser.add_argument(
        "--train_split",
        type=float,
        default=0.80,
        help="Train/validation split inside dataset_size. Default: 0.80."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[100],
        help="Epochs to train. Example: --epochs 50 100 300"
    )

    parser.add_argument(
        "--rays",
        type=int,
        default=32,
        help="Number of StarDist rays. Default: 32."
    )

    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=[4, 4],
        help="StarDist grid. Default: --grid 4 4"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="oyster_0_2mm_stardist",
        help="Base model name."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42."
    )

    parser.add_argument(
        "--convert_binary_to_instances",
        action="store_true",
        help="Use this if masks are binary 0/255 images. Converts connected components to instance labels."
    )

    return parser.parse_args()


# ============================================================
# FILE HANDLING
# ============================================================

def get_files(folder):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.JPG"]
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)


def read_image_rgb(filename):
    """
    Read original image as RGB.
    Output shape: H x W x 3
    """
    with Image.open(filename) as img:
        img = img.convert("RGB")
        return np.array(img)


def read_mask_gray_resize(mask_file, image_shape):
    """
    Read mask as grayscale 2D image.
    Resize mask to match image H x W if needed.
    """
    image_h, image_w = image_shape[:2]

    with Image.open(mask_file) as img:
        img = img.convert("L")

        mask_w, mask_h = img.size

        if (mask_h, mask_w) != (image_h, image_w):
            print("\nResizing annotation:")
            print("Mask file        :", mask_file)
            print("Old mask size    :", (mask_h, mask_w))
            print("Target image size:", (image_h, image_w))

            img = img.resize((image_w, image_h), resample=Image.NEAREST)

        mask = np.array(img)

    return mask


def mask_to_label(mask, convert_binary_to_instances=False):
    """
    Convert mask to StarDist-compatible instance label image.

    StarDist needs:
        background = 0
        each object = unique positive integer label

    Cases:
    1. Binary mask 0/255:
       Convert connected components to labels.
    2. Already labeled mask:
       Keep labels but make sure dtype is int32.
    """

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, but got shape {mask.shape}")

    unique_vals = np.unique(mask)

    # Detect binary masks automatically
    is_binary = len(unique_vals) <= 2

    if convert_binary_to_instances or is_binary:
        binary = mask > 0
        label_mask, n_objects = ndi.label(binary)
        label_mask = label_mask.astype(np.int32)

        return label_mask

    # Otherwise assume mask is already an instance-label image
    label_mask = mask.astype(np.int32)

    # Make sure background is 0
    label_mask[label_mask < 0] = 0

    return label_mask


def print_mask_debug(Y, Y_files, max_print=10):
    print("\nMask sanity check:")
    for i, y in enumerate(Y[:max_print]):
        unique_count = len(np.unique(y))
        print(
            f"{i:03d} | {os.path.basename(Y_files[i])} | "
            f"shape={y.shape} dtype={y.dtype} min={y.min()} max={y.max()} unique={unique_count}"
        )


# ============================================================
# SAVE HELPERS
# ============================================================

def save_split_csv(filename, train_indices, val_indices, test_indices, X_files, Y_files):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "index", "image_file", "mask_file"])

        for idx in train_indices:
            writer.writerow(["train", idx, X_files[idx], Y_files[idx]])

        for idx in val_indices:
            writer.writerow(["validation", idx, X_files[idx], Y_files[idx]])

        for idx in test_indices:
            writer.writerow(["test", idx, X_files[idx], Y_files[idx]])


def save_images_grid(images, filename, title, max_images=25):
    if len(images) == 0:
        print(f"No images to save for {title}")
        return

    images = images[:max_images]

    n = len(images)
    cols = int(math.sqrt(n))
    cols = max(cols, 1)
    rows = int(math.ceil(n / cols))

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(title, fontsize=30)

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = images[i]

        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(np.clip(img, 0, 1))

        ax.axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def save_prediction_images(model, X_data, Y_true, Y_pred, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    n = len(X_data)

    for i in range(n):
        x = X_data[i]
        yt = Y_true[i]
        yp = Y_pred[i]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(np.clip(x, 0, 1))
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(yt, cmap="nipy_spectral")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(yp, cmap="nipy_spectral")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        fig.tight_layout()

        save_path = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_and_save(model, X_data, Y_data, data_type="test"):
    model_dir = os.path.join(model.basedir, model.name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\nPredicting {data_type} images...")

    Y_pred = []

    for x in tqdm(X_data, desc=f"Predicting {data_type}"):
        pred, details = model.predict_instances(
            x,
            n_tiles=model._guess_n_tiles(x),
            show_tile_progress=False
        )
        Y_pred.append(pred.astype(np.int32))

    # Save prediction preview images
    pred_dir = os.path.join(model_dir, f"{data_type}_prediction_images")
    save_prediction_images(model, X_data, Y_data, Y_pred, pred_dir, data_type)

    # Evaluate
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\nEvaluating {data_type} images...")

    stats = [
        matching_dataset(Y_data, Y_pred, thresh=t, show_progress=False)
        for t in tqdm(taus, desc=f"Evaluating {data_type}")
    ]

    # Save CSV
    csv_file = os.path.join(model_dir, f"{data_type}_stats.csv")
    fieldnames = list(stats[0]._asdict().keys())

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in stats:
            writer.writerow(entry._asdict())

    print(f"Saved {data_type} stats CSV: {csv_file}")

    # Save plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    metrics = (
        "precision",
        "recall",
        "accuracy",
        "f1",
        "mean_true_score",
        "mean_matched_score",
        "panoptic_quality",
    )

    counts = ("fp", "tp", "fn")

    for m in metrics:
        ax1.plot(taus, [s._asdict()[m] for s in stats], ".-", lw=2, label=m)

    ax1.set_xlabel(r"IoU threshold $\tau$")
    ax1.set_ylabel("Metric value")
    ax1.grid(True)
    ax1.legend()

    for m in counts:
        ax2.plot(taus, [s._asdict()[m] for s in stats], ".-", lw=2, label=m)

    ax2.set_xlabel(r"IoU threshold $\tau$")
    ax2.set_ylabel("Count")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()

    plot_file = os.path.join(model_dir, f"{data_type}_plots.png")
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)

    print(f"Saved {data_type} plot: {plot_file}")

    # Print useful summary at IoU 0.5
    stat_05 = stats[4]._asdict()

    print(f"\n{data_type.upper()} SUMMARY at IoU 0.5")
    print("n_true   :", stat_05["n_true"])
    print("n_pred   :", stat_05["n_pred"])
    print("tp       :", stat_05["tp"])
    print("fp       :", stat_05["fp"])
    print("fn       :", stat_05["fn"])
    print("precision:", stat_05["precision"])
    print("recall   :", stat_05["recall"])
    print("f1       :", stat_05["f1"])
    print("accuracy :", stat_05["accuracy"])

    return stats, Y_pred


# ============================================================
# MAIN
# ============================================================

def main(args):

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load filenames
    # ------------------------------------------------------------

    X_files = get_files(args.image_dir)
    Y_files = get_files(args.mask_dir)

    print("Image folder:", args.image_dir)
    print("Mask folder :", args.mask_dir)
    print("Number of image files:", len(X_files))
    print("Number of mask files :", len(Y_files))

    if len(X_files) == 0:
        raise ValueError(f"No image files found in {args.image_dir}")

    if len(Y_files) == 0:
        raise ValueError(f"No mask files found in {args.mask_dir}")

    if len(X_files) != len(Y_files):
        raise ValueError(
            f"Image/mask count mismatch. Images={len(X_files)}, Masks={len(Y_files)}. "
            "Make sure each image has one matching annotation."
        )

    print("\nFirst 10 image/mask pairs:")
    for xf, yf in list(zip(X_files, Y_files))[:10]:
        print(os.path.basename(xf), " ---> ", os.path.basename(yf))

    # ------------------------------------------------------------
    # Load images and masks
    # ------------------------------------------------------------

    print("\nLoading images...")
    X_raw = [read_image_rgb(xf) for xf in tqdm(X_files, desc="Loading images")]

    print("\nLoading masks...")
    Y_raw = []

    for yf, x in tqdm(list(zip(Y_files, X_raw)), desc="Loading masks"):
        y = read_mask_gray_resize(yf, x.shape)
        y = mask_to_label(
            y,
            convert_binary_to_instances=args.convert_binary_to_instances
        )
        y = fill_label_holes(y)
        Y_raw.append(y.astype(np.int32))

    # ------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------

    print("\nChecking image/mask shapes...")

    bad_pairs = []

    for i, (x, y, xf, yf) in enumerate(zip(X_raw, Y_raw, X_files, Y_files)):
        if x.shape[:2] != y.shape[:2]:
            bad_pairs.append((i, x.shape, y.shape, xf, yf))

    if bad_pairs:
        print("\nERROR: Some images and masks still do not match.")
        for item in bad_pairs[:10]:
            i, xshape, yshape, xf, yf = item
            print("-" * 80)
            print("Index:", i)
            print("Image shape:", xshape)
            print("Mask shape :", yshape)
            print("Image file :", xf)
            print("Mask file  :", yf)

        raise ValueError("Images and masks must have matching height and width.")

    print("All image/mask shapes match.")

    print_mask_debug(Y_raw, Y_files, max_print=10)

    # ------------------------------------------------------------
    # Normalize images
    # ------------------------------------------------------------

    print("\nNormalizing images...")
    axis_norm = (0, 1)
    X = [
        normalize(x, 1, 99.8, axis=axis_norm).astype(np.float32)
        for x in tqdm(X_raw, desc="Normalizing")
    ]

    Y = Y_raw

    total_data = len(X)

    # ------------------------------------------------------------
    # Split data
    # ------------------------------------------------------------

    if args.testing_size < 1:
        raise ValueError("testing_size must be at least 1.")

    if args.testing_size >= total_data:
        raise ValueError(
            f"testing_size={args.testing_size} must be smaller than total images={total_data}."
        )

    if args.dataset_size is None:
        dataset_size = total_data - args.testing_size
    else:
        dataset_size = args.dataset_size

    if dataset_size < 2:
        raise ValueError("dataset_size must be at least 2 so train/validation split can work.")

    if dataset_size > total_data - args.testing_size:
        raise ValueError(
            f"dataset_size={dataset_size} is too large. "
            f"It must be <= total_data - testing_size = {total_data - args.testing_size}"
        )

    if not (0 < args.train_split < 1):
        raise ValueError("train_split must be between 0 and 1.")

    all_indices = np.arange(total_data)

    test_indices = rng.choice(all_indices, size=args.testing_size, replace=False)

    remaining_indices = np.array(
        [i for i in all_indices if i not in set(test_indices)]
    )

    selected_indices = rng.choice(
        remaining_indices,
        size=dataset_size,
        replace=False
    )

    rng.shuffle(selected_indices)

    n_train = int(args.train_split * dataset_size)

    if n_train < 1:
        raise ValueError("Training set is empty. Increase dataset_size or train_split.")

    if dataset_size - n_train < 1:
        raise ValueError("Validation set is empty. Increase dataset_size or reduce train_split.")

    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:]

    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]

    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]

    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    print("\nDataset split:")
    print("Total images       :", total_data)
    print("Train+Val size     :", dataset_size)
    print("Training set size  :", len(X_train))
    print("Validation set size:", len(X_val))
    print("Testing set size   :", len(X_test))

    # ------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------

    dataset_dir = os.path.join(args.output_dir, f"datasize_{dataset_size}")
    os.makedirs(dataset_dir, exist_ok=True)

    split_csv = os.path.join(dataset_dir, "train_val_test_split.csv")
    save_split_csv(
        split_csv,
        train_indices,
        val_indices,
        test_indices,
        X_files,
        Y_files
    )

    print("Saved split CSV:", split_csv)

    save_images_grid(
        X_train,
        os.path.join(dataset_dir, "training_images.png"),
        "Training Images"
    )

    save_images_grid(
        X_val,
        os.path.join(dataset_dir, "validation_images.png"),
        "Validation Images"
    )

    save_images_grid(
        X_test,
        os.path.join(dataset_dir, "testing_images.png"),
        "Testing Images"
    )

    save_images_grid(
        Y_train,
        os.path.join(dataset_dir, "training_masks.png"),
        "Training Masks"
    )

    save_images_grid(
        Y_val,
        os.path.join(dataset_dir, "validation_masks.png"),
        "Validation Masks"
    )

    save_images_grid(
        Y_test,
        os.path.join(dataset_dir, "testing_masks.png"),
        "Testing Masks"
    )

    # ------------------------------------------------------------
    # StarDist config
    # ------------------------------------------------------------

    n_channel = 3
    n_rays = args.rays
    grid = tuple(args.grid)

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        n_channel_in=n_channel,
        train_patch_size=(256, 256),
        train_batch_size=2,
    )

    print("\nStarDist config:")
    print(conf)

    # ------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------

    def random_fliprot(img, mask):
        assert img.ndim >= mask.ndim

        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))

        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
        mask = mask.transpose(perm)

        for ax in axes:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)

        return img, mask

    def random_intensity_change(img):
        img = img * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)
        return img

    def augmenter(x, y):
        x, y = random_fliprot(x, y)

        x = random_intensity_change(x)

        sig = 0.02 * np.random.uniform(0, 1)
        x = x + sig * np.random.normal(0, 1, x.shape)

        x = np.clip(x, 0, 1)

        return x, y

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    for epoch_count in args.epochs:

        model_name = (
            f"{args.model_name}_datasize_{dataset_size}"
            f"_test_{args.testing_size}"
            f"_rays_{n_rays}"
            f"_grid_{grid[0]}x{grid[1]}"
            f"_epochs_{epoch_count}"
        )

        print("\n" + "=" * 80)
        print("Training model:", model_name)
        print("=" * 80)

        model = StarDist2D(
            conf,
            name=model_name,
            basedir=dataset_dir
        )

        median_size = np.array(calculate_extents(list(Y_train), np.median))[:2]
        fov = np.array(model._axes_tile_overlap("YX"))[:2]

        print("Median object size     :", median_size)
        print("Network field of view  :", fov)

        if any(median_size > fov):
            print("\nWARNING:")
            print("Median object size is larger than the network field of view.")
            print("Try increasing grid, for example: --grid 8 8")
            print("Or use larger train_patch_size if memory allows.\n")

        model.train(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            augmenter=augmenter,
            epochs=epoch_count
        )

        print("\nOptimizing thresholds on validation set...")
        model.optimize_thresholds(X_val, Y_val)

        print("\nEvaluating validation set...")
        evaluate_and_save(model, X_val, Y_val, data_type="validation")

        print("\nEvaluating test set...")
        evaluate_and_save(model, X_test, Y_test, data_type="test")

    print("\nTraining is complete.")
    print("All results saved in:", args.output_dir)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    main(args)