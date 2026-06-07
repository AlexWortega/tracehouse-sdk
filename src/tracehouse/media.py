"""Media objects for run logging — wandb-style ``cm.Image`` / ``cm.Video``.

Zero hard deps: accepts a file path or raw ``bytes`` out of the box. PIL images
and numpy arrays work too *if* Pillow is installed (lazy-imported only then).

    import tracehouse as cm
    run = cm.init_run(project="demo", name="sft")
    run.log({"samples": cm.Image("out/epoch3.png", caption="epoch 3")}, step=3)
    run.log_video("rollout", "rollout.mp4")
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Optional, Tuple

_IMAGE_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}
_VIDEO_EXT_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


def _sniff_image_mime(b: bytes) -> Optional[str]:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return None


def _looks_like_path(data: Any) -> bool:
    return isinstance(data, (str, Path)) and len(str(data)) < 4096 and os.path.exists(str(data))


def _coerce_image(data: Any, fmt: Optional[str]) -> Tuple[bytes, str]:
    if isinstance(data, (bytes, bytearray)):
        b = bytes(data)
        return b, (f"image/{fmt.lower()}" if fmt else None) or _sniff_image_mime(b) or "image/png"
    if _looks_like_path(data):
        p = Path(data)
        b = p.read_bytes()
        return b, _IMAGE_EXT_MIME.get(p.suffix.lower()) or _sniff_image_mime(b) or "image/png"
    # PIL Image — has .save(buf, format=...)
    if hasattr(data, "save") and hasattr(data, "size"):
        buf = io.BytesIO()
        f = (fmt or "PNG").upper()
        data.save(buf, format=f)
        return buf.getvalue(), f"image/{f.lower().replace('jpg', 'jpeg')}"
    # numpy array → PNG via Pillow (only path that needs a dep)
    if hasattr(data, "__array_interface__") or type(data).__module__ == "numpy":
        try:
            import numpy as _np  # type: ignore
            from PIL import Image as _PILImage  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise TypeError(
                "logging a numpy array as an image needs Pillow + numpy "
                f"(`pip install pillow numpy`); or pass a file path / PNG bytes. ({e})"
            )
        arr = _np.asarray(data)
        if arr.dtype != _np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            span = (hi - lo) or 1.0
            arr = (255.0 * (arr - lo) / span).astype("uint8")
        buf = io.BytesIO()
        _PILImage.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    raise TypeError(
        f"cm.Image: unsupported data {type(data)!r} — pass a file path, raw bytes, "
        "a PIL Image, or a numpy array."
    )


def _coerce_video(data: Any, fmt: Optional[str]) -> Tuple[bytes, str]:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data), (f"video/{fmt.lower()}" if fmt else None) or "video/mp4"
    if _looks_like_path(data):
        p = Path(data)
        return p.read_bytes(), _VIDEO_EXT_MIME.get(p.suffix.lower()) or "video/mp4"
    raise TypeError("cm.Video: pass a file path or raw bytes.")


class Image:
    """An image to log to a run. ``data`` is a file path, raw bytes, a PIL
    Image, or a numpy array (HxW or HxWxC; needs Pillow)."""

    media_type = "image"

    def __init__(self, data: Any, *, caption: Optional[str] = None, format: Optional[str] = None) -> None:
        self.bytes, self.mime = _coerce_image(data, format)
        self.caption = caption


class Video:
    """A video to log to a run. ``data`` is a file path or raw bytes
    (mp4 / webm / mov / mkv)."""

    media_type = "video"

    def __init__(self, data: Any, *, caption: Optional[str] = None, format: Optional[str] = None) -> None:
        self.bytes, self.mime = _coerce_video(data, format)
        self.caption = caption
