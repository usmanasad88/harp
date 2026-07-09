"""Face detection for head tracking — RealSense (with depth) or plain webcam.

Port of harpcontrol's depth_detect.py minus ROS: the same YOLOv8n-face ONNX
model (assets/models/), letterbox → decode → NMS pipeline, and depth-based
nearest-face selection. The camera is pluggable per PLAN.md: RealSense gives
color + aligned depth so the *nearest* face is tracked among several; a plain
webcam still drives tracking, falling back to the *largest* face box.

pyrealsense2 is imported lazily so webcam mode works even where the RealSense
SDK is missing or broken.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "models" / "yolov8n-face-lindevs.onnx"
)

FRAME_W = 640
FRAME_H = 480
DEPTH_W = 640
DEPTH_H = 360
FPS = 30

ONNX_SIZE = 640
CONF_THRESH = 0.35
NMS_IOU_THRESH = 0.5


# ----------------------------------------------------------------------
# Detection pipeline (ported from depth_detect.py)
# ----------------------------------------------------------------------

def _letterbox(img, new_shape=ONNX_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, (left, top)


def _iou(a, b):
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _nms(boxes, scores, iou_th=NMS_IOU_THRESH):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        idxs = np.array([j for j in idxs[1:] if _iou(boxes[i], boxes[j]) < iou_th])
    return keep


def _decode(output, conf_th=CONF_THRESH):
    out = output[0]
    if out.ndim == 3:
        out = out[0]
    out = out.transpose(1, 0)

    boxes = []
    scores = []
    for (x, y, w, h, conf) in out:
        conf = float(conf)
        if conf < conf_th:
            continue
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        scores.append(conf)
    return boxes, scores


def _scale_box(b, ratio, pad, h, w):
    left, top = pad
    x1 = int((b[0] - left) / ratio)
    y1 = int((b[1] - top) / ratio)
    x2 = int((b[2] - left) / ratio)
    y2 = int((b[3] - top) / ratio)
    return [
        max(0, min(w - 1, x1)),
        max(0, min(h - 1, y1)),
        max(0, min(w - 1, x2)),
        max(0, min(h - 1, y2)),
    ]


class FaceDetector:
    """YOLOv8n-face ONNX (CPU) → list of (x1, y1, x2, y2) boxes, NMS applied."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"face model missing: {model_path}")
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[list[int]]:
        h, w = frame.shape[:2]
        padded, ratio, pad = _letterbox(frame)
        inp = padded.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :]

        outputs = self._sess.run(None, {self._input_name: inp})
        boxes_padded, scores = _decode(outputs[0])
        boxes = [_scale_box(b, ratio, pad, h, w) for b in boxes_padded]
        keep = _nms(boxes, scores)
        return [boxes[i] for i in keep]


def pick_face(
    boxes: list[list[int]], depth_frame
) -> tuple[int, int, float | None] | None:
    """Choose the face to track → (cx, cy, depth_m | None).

    With a depth frame: the nearest face by median depth over a 5×5 patch at
    the box center (depth_detect.py's prioritization). Without: the largest box.
    """
    if not boxes:
        return None

    if depth_frame is None:
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return (x1 + x2) // 2, (y1 + y2) // 2, None

    w, h = depth_frame.get_width(), depth_frame.get_height()
    nearest = None
    nearest_depth = float("inf")
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        patch = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    d = depth_frame.get_distance(px, py)
                    if d > 0:
                        patch.append(d)
        dval = float(np.median(patch)) if patch else float("inf")
        if dval < nearest_depth:
            nearest_depth = dval
            nearest = (cx, cy)
    if nearest is None:
        return None
    return nearest[0], nearest[1], (None if nearest_depth == float("inf") else nearest_depth)


# ----------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------

class RealSenseCamera:
    """Color + aligned depth. `read()` → (bgr_frame | None, depth_frame | None)."""

    name = "realsense"

    # (color_w, color_h, depth_w, depth_h, fps) — tried in order. The first is
    # the full-rate profile; it only resolves on a USB 3 link. Over USB 2 (a
    # USB 2 hub/cable — the camera reports "2.1") the D435 caps at 15 fps, so
    # the second profile keeps face tracking alive, just at half rate.
    _PROFILES = (
        (FRAME_W, FRAME_H, DEPTH_W, DEPTH_H, FPS),
        (FRAME_W, FRAME_H, FRAME_W, FRAME_H, 15),
    )

    def __init__(self) -> None:
        import pyrealsense2 as rs  # lazy: webcam mode must not need the SDK

        self._rs = rs
        self._pipeline = rs.pipeline()
        last_error: Exception | None = None
        for cw, ch, dw, dh, fps in self._PROFILES:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
            try:
                profile = self._pipeline.start(cfg)
            except RuntimeError as exc:
                last_error = exc
                continue
            device = profile.get_device()
            usb = (
                device.get_info(rs.camera_info.usb_type_descriptor)
                if device.supports(rs.camera_info.usb_type_descriptor)
                else "?"
            )
            logger.info(
                "camera: RealSense on USB %s — color %dx%d@%d, depth %dx%d@%d",
                usb, cw, ch, fps, dw, dh, fps,
            )
            if usb.startswith("2"):
                logger.warning(
                    "camera: RealSense is on a USB 2 link (hub/cable?) — running "
                    "at %d fps; plug it into a USB 3 port for full rate", fps
                )
            break
        else:
            raise RuntimeError(f"no RealSense stream profile resolved: {last_error}")
        self._align = rs.align(rs.stream.color)

    def read(self):
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color:
            return None, None
        return np.asanyarray(color.get_data()), (depth if depth else None)

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


class WebcamCamera:
    """Plain cv2 webcam — no depth, tracking falls back to the largest face."""

    name = "webcam"

    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam {index}")

    def read(self):
        ok, frame = self._cap.read()
        return (frame if ok else None), None

    def close(self) -> None:
        self._cap.release()


def open_camera(backend: str = "auto", webcam_index: int = 0):
    """`auto` tries RealSense first, then webcam. Returns None if none opens."""
    if backend in ("auto", "realsense"):
        try:
            cam = RealSenseCamera()
            logger.info("camera: RealSense (color + depth)")
            return cam
        except Exception as exc:
            level = logging.WARNING if backend == "realsense" else logging.INFO
            logger.log(level, "camera: RealSense unavailable (%s)", exc)
            if backend == "realsense":
                return None
    if backend in ("auto", "webcam"):
        try:
            cam = WebcamCamera(webcam_index)
            logger.info("camera: webcam %d (no depth)", webcam_index)
            return cam
        except Exception as exc:
            logger.warning("camera: webcam unavailable (%s)", exc)
    return None
