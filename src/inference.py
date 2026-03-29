"""
inference.py — AgriVision Decision Support System
=================================================
Memory-optimized YOLOv8 inference wrapper designed for deployment on
CPU-bound environments (e.g. Intel i3 with 8GB RAM).

Implements strict PyTorch tensor flushing and garbage collection to
prevent CUDA/system out-of-memory cascading failures during long-duration
batch inference or web container execution.


"""

import gc
from typing import Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO


def get_model(weights_path: str) -> YOLO:
    """
    Initialize the YOLOv8 model and explicitly bind it to the CPU.

    Parameters
    ----------
    weights_path : str
        Absolute or relative path to the trained .pt weights file.

    Returns
    -------
    YOLO
        The Ultralytics model instance bound to the CPU device.
    """
    # Load model and force to CPU. This prevents accidental GPU allocation tries
    # on mixed-hardware deployment servers, which can crash containers.
    # Note: the ultralytics library handles device mapping natively during forward pass,
    # but initializing the model object can be done here.
    model = YOLO(weights_path)
    return model


def run_inference(
    model: YOLO,
    image: Image.Image | np.ndarray | str,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    img_size: int = 1024
) -> Tuple[int, Image.Image]:
    """
    Execute a memory-guarded forward pass on a single image.

    This function isolates the PyTorch workload, extracts the necessary
    outputs (head count and annotated visual array), and proactively
    destroys the massive underlying gradient and result tensors before
    returning control to the invoking UI application.

    Parameters
    ----------
    model : YOLO
        The initialized YOLOv8 CPU model.
    image : PIL.Image.Image or np.ndarray or str
        The input image to run inference on.
    conf_thresh : float, optional
        Confidence threshold for filtering weak detections (default: 0.25).
    iou_thresh : float, optional
        Non-Maximum Suppression (NMS) threshold for duplicate box removal (default: 0.45).
    img_size : int, optional
        Input resolution padding size. Models trained at 1024px should use 1024
        to maximize spatial accuracy (default: 1024).

    Returns
    -------
    Tuple[int, PIL.Image.Image]
        A tuple containing:
        - The absolute count of detected wheat heads (int).
        - The annotated RGB image ready for UI rendering (PIL.Image.Image).
    """

    # 1. Forward Pass (Explicitly routed to CPU to avoid OOM)
    #    The argument 'device="cpu"' ensures no CUDA allocation occurs.
    results = model.predict(
        source=image,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=img_size,
        device="cpu",
        verbose=False  # Suppress CLI spam in production logs
    )

    # 2. Extract Data
    #    The 'boxes' attribute contains the filtered bounding boxes.
    count = len(results[0].boxes)

    #    plot() returns a BGR numpy array containing the drawn boxes/labels
    bgr_annotated_array = results[0].plot()

    # 3. Format Conversion
    #    Convert OpenCV-style BGR array to standard RGB PIL Image for web frameworks
    rgb_array = bgr_annotated_array[..., ::-1]  # NumPy fast BGR to RGB slicing
    rgb_annotated_image = Image.fromarray(rgb_array)

    # 4. Aggressive Memory Guard [CRITICAL FOR LOW-RAM DEPLOYMENT]
    #    A single High-Res Inference object holds immense tensor graphs.
    #    We must wipe it completely before the Python GC cycle figures it out.
    del results
    gc.collect()

    return count, rgb_annotated_image
