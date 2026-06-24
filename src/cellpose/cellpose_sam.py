#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full Cellpose batch segmentation script.

Important:
This version saves TIFF masks, PNG overlays, and PNG outlines.
It does NOT save PNG masks.

Outputs:
1. masks_tif/
   - integer label masks as TIFF
   - same H x W size as original image

2. overlays_png/
   - colored mask overlay on original image
   - same H x W size as original image

3. outlines_png/
   - red object boundaries on original image
   - same H x W size as original image

4. per_object_csv/
   - per-object area, centroid, bounding box

5. cellpose_summary.csv
   - object count and area summary per image

Run:

    python cellpose_png_overlay_only.py

To change paths, edit DEFAULT_INPUT_DIR and DEFAULT_OUTPUT_DIR below.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import tifffile

from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries

from cellpose import models


# ============================================================
# DEFAULT SETTINGS
# Change only these when needed
# ============================================================

DEFAULT_INPUT_DIR = "/scratch/gsunka1/LSG_0-2mm/original"
DEFAULT_OUTPUT_DIR = "/scratch/gsunka1/LSG_0-2mm/cellpose_outputs"

DEFAULT_MODEL = "cpsam_v2"
DEFAULT_DIAMETER = 30
DEFAULT_CHANNEL_AXIS = -1
DEFAULT_CELLPROB_THRESHOLD = -0.5
DEFAULT_FLOW_THRESHOLD = 0.4
DEFAULT_MIN_SIZE = 5
DEFAULT_USE_GPU = True


# ============================================================
# IMAGE READING
# ============================================================

def read_image(path):
    """
    Read image as numpy array.

    Supports:
    - PNG
    - JPG/JPEG
    - TIF/TIFF

    Returns:
    - grayscale: H x W
    - RGB: H x W x 3
    - TIFF stack: depends on file structure
    """

    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in [".tif", ".tiff"]:
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Could not read image: {path}")

        # OpenCV reads BGR, convert to RGB
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Remove alpha channel if present
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    return img


def get_original_hw(img):
    """
    Return original height and width.

    Handles:
    - H x W
    - H x W x C
    - Z x H x W
    - Z x H x W x C
    """

    if img.ndim == 2:
        return img.shape[0], img.shape[1]

    if img.ndim == 3:
        # RGB image: H x W x C
        if img.shape[-1] in [1, 3, 4]:
            return img.shape[0], img.shape[1]

        # Z-stack: Z x H x W
        return img.shape[1], img.shape[2]

    if img.ndim == 4:
        # Z x H x W x C
        return img.shape[1], img.shape[2]

    raise ValueError(f"Unsupported image shape: {img.shape}")


# ============================================================
# SIZE FIXING FUNCTIONS
# ============================================================

def resize_mask_to_original(masks, original_hw):
    """
    Resize mask back to original image size.

    Uses nearest-neighbor interpolation so label IDs remain unchanged.
    """

    original_h, original_w = original_hw

    if masks.ndim == 2:
        if masks.shape == (original_h, original_w):
            return masks.astype(np.uint32)

        resized = cv2.resize(
            masks.astype(np.uint32),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return resized.astype(np.uint32)

    if masks.ndim == 3:
        # Z x H x W
        if masks.shape[1:3] == (original_h, original_w):
            return masks.astype(np.uint32)

        resized_stack = []

        for z in range(masks.shape[0]):
            resized_z = cv2.resize(
                masks[z].astype(np.uint32),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )
            resized_stack.append(resized_z.astype(np.uint32))

        return np.stack(resized_stack, axis=0)

    raise ValueError(f"Unsupported mask shape: {masks.shape}")


def force_rgb_to_original_size(img_rgb, original_hw):
    """
    Force overlay or outline image to original H x W size.
    """

    original_h, original_w = original_hw

    if img_rgb.shape[:2] == (original_h, original_w):
        return img_rgb

    resized = cv2.resize(
        img_rgb,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    return resized


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def normalize_for_display(img):
    """
    Convert image to uint8 RGB for visualization.
    """

    arr = img.copy()

    # If 4D stack, show middle Z plane
    if arr.ndim == 4:
        z = arr.shape[0] // 2
        arr = arr[z]

    # If 3D and looks like Z x H x W, show middle Z plane
    if arr.ndim == 3 and arr.shape[-1] not in [1, 3, 4]:
        z = arr.shape[0] // 2
        arr = arr[z]

    # If grayscale, convert to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # If single channel, repeat to RGB
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # If more than 3 channels, keep first 3
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)

    p1, p99 = np.percentile(arr, (1, 99))

    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    else:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()

    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    return arr


def get_2d_mask_for_display(masks):
    """
    Convert mask to 2D for overlay/outline display.
    """

    if masks.ndim == 2:
        return masks

    if masks.ndim == 3:
        z = masks.shape[0] // 2
        return masks[z]

    raise ValueError(f"Unsupported mask shape for display: {masks.shape}")


def make_mask_overlay(img, masks, alpha=0.45):
    """
    Create colored mask overlay on image.
    """

    base = normalize_for_display(img)
    masks_show = get_2d_mask_for_display(masks)

    # Safety resize before overlay
    if masks_show.shape != base.shape[:2]:
        masks_show = cv2.resize(
            masks_show.astype(np.uint32),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint32)

    overlay = base.copy()
    rng = np.random.default_rng(42)

    labels = np.unique(masks_show)
    labels = labels[labels != 0]

    color_layer = np.zeros_like(base, dtype=np.uint8)

    for lab in labels:
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        color_layer[masks_show == lab] = color

    mask_pixels = masks_show > 0

    overlay[mask_pixels] = (
        (1 - alpha) * base[mask_pixels]
        + alpha * color_layer[mask_pixels]
    ).astype(np.uint8)

    return overlay


def make_outline_image(img, masks):
    """
    Draw red boundaries around masks.
    """

    base = normalize_for_display(img)
    masks_show = get_2d_mask_for_display(masks)

    # Safety resize before drawing boundaries
    if masks_show.shape != base.shape[:2]:
        masks_show = cv2.resize(
            masks_show.astype(np.uint32),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint32)

    boundaries = find_boundaries(masks_show, mode="outer")

    out = base.copy()
    out[boundaries] = [255, 0, 0]

    return out


def save_png_rgb(path, img):
    """
    Save RGB image as PNG using OpenCV.
    """

    path = str(path)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape: {img.shape}")

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def save_mask_tif(path, masks):
    """
    Save Cellpose label mask as TIFF.

    Background = 0
    Oyster 1 = 1
    Oyster 2 = 2
    Oyster 3 = 3
    ...

    TIFF keeps integer labels safely.
    """

    path = str(path)

    masks_to_save = masks.astype(np.uint32)

    tifffile.imwrite(path, masks_to_save)


# ============================================================
# MASK SUMMARY FUNCTIONS
# ============================================================

def summarize_masks(masks, image_name):
    """
    Compute object count and area statistics.
    """

    if masks.ndim == 3:
        labels = np.unique(masks)
        labels = labels[labels != 0]

        areas = []

        for lab in labels:
            areas.append(int(np.sum(masks == lab)))

        if len(areas) == 0:
            return {
                "image": image_name,
                "num_objects": 0,
                "mean_area_px": 0,
                "median_area_px": 0,
                "min_area_px": 0,
                "max_area_px": 0,
            }

        return {
            "image": image_name,
            "num_objects": int(len(areas)),
            "mean_area_px": float(np.mean(areas)),
            "median_area_px": float(np.median(areas)),
            "min_area_px": int(np.min(areas)),
            "max_area_px": int(np.max(areas)),
        }

    props = regionprops_table(
        masks.astype(np.int32),
        properties=["label", "area", "centroid", "bbox"]
    )

    if len(props["label"]) == 0:
        return {
            "image": image_name,
            "num_objects": 0,
            "mean_area_px": 0,
            "median_area_px": 0,
            "min_area_px": 0,
            "max_area_px": 0,
        }

    areas = np.asarray(props["area"], dtype=float)

    return {
        "image": image_name,
        "num_objects": int(len(areas)),
        "mean_area_px": float(np.mean(areas)),
        "median_area_px": float(np.median(areas)),
        "min_area_px": float(np.min(areas)),
        "max_area_px": float(np.max(areas)),
    }


def save_per_object_csv(masks, output_csv):
    """
    Save object-level measurements.
    """

    if masks.ndim == 3:
        rows = []
        labels = np.unique(masks)
        labels = labels[labels != 0]

        for lab in labels:
            coords = np.argwhere(masks == lab)

            area = coords.shape[0]

            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0)

            rows.append({
                "label": int(lab),
                "area_voxels": int(area),
                "centroid_z": float(coords[:, 0].mean()),
                "centroid_y": float(coords[:, 1].mean()),
                "centroid_x": float(coords[:, 2].mean()),
                "bbox_zmin": int(zmin),
                "bbox_ymin": int(ymin),
                "bbox_xmin": int(xmin),
                "bbox_zmax": int(zmax),
                "bbox_ymax": int(ymax),
                "bbox_xmax": int(xmax),
            })

        pd.DataFrame(rows).to_csv(output_csv, index=False)
        return

    props = regionprops_table(
        masks.astype(np.int32),
        properties=[
            "label",
            "area",
            "centroid",
            "bbox"
        ]
    )

    pd.DataFrame(props).to_csv(output_csv, index=False)


# ============================================================
# MAIN SEGMENTATION
# ============================================================

def run_segmentation(args):

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    mask_dir = output_dir / "masks_tif"
    overlay_dir = output_dir / "overlays_png"
    outline_dir = output_dir / "outlines_png"
    object_csv_dir = output_dir / "per_object_csv"

    for folder in [mask_dir, overlay_dir, outline_dir, object_csv_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    image_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    image_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in image_exts
    ])

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {input_dir}")

    print("=" * 80)
    print(f"Found {len(image_paths)} images")
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print("=" * 80)

    print(f"Loading Cellpose model: {args.model}")

    model = models.CellposeModel(
        gpu=args.gpu,
        pretrained_model=args.model
    )

    summary_rows = []

    for idx, image_path in enumerate(image_paths, start=1):

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")

        img = read_image(image_path)

        original_hw = get_original_hw(img)
        original_h, original_w = original_hw

        print(f"Original image shape: {img.shape}")
        print(f"Original H x W    : {original_h} x {original_w}")

        masks, flows, styles = model.eval(
            img,
            diameter=args.diameter,
            batch_size=args.batch_size,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            do_3D=args.do_3D,
            anisotropy=args.anisotropy,
            flow3D_smooth=args.flow3D_smooth,
            stitch_threshold=args.stitch_threshold,
            channel_axis=args.channel_axis,
            z_axis=args.z_axis,
            normalize=True,
            compute_masks=True,
        )

        print(f"Raw Cellpose mask shape: {masks.shape}")

        # Resize mask back to original image H x W
        masks = resize_mask_to_original(masks, original_hw)

        print(f"Final saved mask shape: {masks.shape}")

        stem = image_path.stem

        mask_path = mask_dir / f"{stem}_masks.tif"
        overlay_path = overlay_dir / f"{stem}_overlay.png"
        outline_path = outline_dir / f"{stem}_outline.png"
        object_csv_path = object_csv_dir / f"{stem}_objects.csv"

        # Save label mask as TIFF
        save_mask_tif(mask_path, masks)

        # Make overlay and outline
        overlay = make_mask_overlay(img, masks)
        outline = make_outline_image(img, masks)

        # Force overlay and outline to original size
        overlay = force_rgb_to_original_size(overlay, original_hw)
        outline = force_rgb_to_original_size(outline, original_hw)

        print(f"Overlay saved size: {overlay.shape[:2]}")
        print(f"Outline saved size: {outline.shape[:2]}")

        # Save overlay and outline as PNG
        save_png_rgb(overlay_path, overlay)
        save_png_rgb(outline_path, outline)

        # Save per-object measurements
        save_per_object_csv(masks, object_csv_path)

        # Save image-level summary
        row = summarize_masks(masks, image_path.name)

        row["original_height"] = original_h
        row["original_width"] = original_w
        row["mask_shape"] = str(masks.shape)
        row["overlay_shape"] = str(overlay.shape)
        row["outline_shape"] = str(outline.shape)
        row["mask_file"] = str(mask_path)
        row["overlay_file"] = str(overlay_path)
        row["outline_file"] = str(outline_path)
        row["object_csv"] = str(object_csv_path)

        summary_rows.append(row)

        print(f"Objects found: {row['num_objects']}")
        print(f"Saved TIFF mask: {mask_path}")
        print(f"Saved overlay  : {overlay_path}")
        print(f"Saved outline  : {outline_path}")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = output_dir / "cellpose_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("Segmentation finished.")
    print(f"Summary CSV saved to: {summary_csv}")
    print("=" * 80)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Batch Cellpose segmentation with TIFF masks and PNG overlays."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing input images."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where outputs will be saved."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Pretrained Cellpose model name or path to custom model. "
            "Examples: cpsam_v2, cpsam, cpdino, cpdino-vitb, or /path/to/custom_model"
        )
    )

    parser.add_argument(
        "--diameter",
        type=float,
        default=DEFAULT_DIAMETER,
        help=(
            "Approximate object diameter in pixels. "
            "For oyster 0-2 mm images, try 20, 30, 40, 50."
        )
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        default=DEFAULT_USE_GPU,
        help="Use GPU if available."
    )

    parser.add_argument(
        "--no_gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU and run on CPU."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for Cellpose tiled inference."
    )

    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=DEFAULT_FLOW_THRESHOLD,
        help=(
            "Flow error threshold. "
            "Higher keeps more masks; lower removes more masks."
        )
    )

    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=DEFAULT_CELLPROB_THRESHOLD,
        help=(
            "Cell probability threshold. "
            "Lower finds more objects; higher reduces false positives."
        )
    )

    parser.add_argument(
        "--min_size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help="Remove masks smaller than this pixel area."
    )

    parser.add_argument(
        "--channel_axis",
        type=int,
        default=DEFAULT_CHANNEL_AXIS,
        help=(
            "Channel axis. "
            "For H x W x C RGB images, usually use -1. "
            "If unsure, leave as None."
        )
    )

    parser.add_argument(
        "--z_axis",
        type=int,
        default=None,
        help="Z axis for stacks. Usually None for 2D images."
    )

    parser.add_argument(
        "--do_3D",
        action="store_true",
        help="Run true 3D Cellpose segmentation."
    )

    parser.add_argument(
        "--anisotropy",
        type=float,
        default=None,
        help="3D anisotropy factor. Example: 2.0 if Z spacing is twice XY spacing."
    )

    parser.add_argument(
        "--flow3D_smooth",
        type=float,
        default=0,
        help="Smooth 3D flows. Example: 2."
    )

    parser.add_argument(
        "--stitch_threshold",
        type=float,
        default=0.0,
        help=(
            "For 2D plane-by-plane segmentation, stitch masks into 3D if > 0. "
            "Keep 0.0 for normal 2D images."
        )
    )

    return parser.parse_args()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    run_segmentation(args)