from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import (
    FORMATS_HELP_MSG,
    HELP_URL,
    IMG_FORMATS,
    exif_size,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_child_by_tag(node: Optional[ET.Element], tag: str) -> Optional[ET.Element]:
    if node is None:
        return None
    for child in list(node):
        if isinstance(child.tag, str) and _local_tag(child.tag) == tag:
            return child
    return None


def _find_text(node: Optional[ET.Element], tag: str, recursive: bool = False) -> str:
    if node is None:
        return ""
    elements = node.iter() if recursive else list(node)
    for elem in elements:
        if not isinstance(elem.tag, str):
            continue
        if _local_tag(elem.tag) != tag:
            continue
        text = (elem.text or "").strip()
        if text:
            return text
    return ""


def _to_float(text: str) -> Optional[float]:
    value = (text or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _looks_like_xml_file(label_file: str) -> bool:
    """Best-effort check to avoid parsing non-XML labels with XML parser."""
    try:
        with open(label_file, "rb") as f:
            prefix = f.read(256).lstrip()
    except OSError:
        return False

    if not prefix:
        return False
    return prefix.startswith(b"<?xml") or prefix.startswith(b"<annotation") or prefix.startswith(b"<")


def _parse_yolo_rows_fallback(label_file: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Fallback parser for YOLO txt labels that may be mislabeled with .xml suffix."""
    rows: List[List[float]] = []
    warnings: List[str] = []

    try:
        with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                values = [_to_float(x) for x in parts]
                if any(v is None for v in values):
                    continue

                cls, cx, cy, bw, bh = [float(v) for v in values if v is not None]
                if cls < 0:
                    warnings.append(f"line {line_no}: negative class id")
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    warnings.append(f"line {line_no}: non-normalized yolo bbox")
                    continue

                rows.append([cls, cx, cy, bw, bh])
    except OSError as e:
        return np.zeros((0, 5), dtype=np.float32), [], [f"fallback txt parse failed: {e}"]

    if not rows:
        return np.zeros((0, 5), dtype=np.float32), [], warnings

    labels = np.asarray(rows, dtype=np.float32)
    return labels, ["" for _ in range(len(labels))], warnings


def _parse_xml_rows(
    xml_file: str,
    name_to_id: Dict[str, int],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, List[str], List[str]]:
    h, w = int(shape[0]), int(shape[1])
    if h <= 1 or w <= 1:
        return np.zeros((0, 5), dtype=np.float32), [], ["invalid image shape"]

    rows: List[List[float]] = []
    descriptions: List[str] = []
    warnings: List[str] = []

    try:
        root = ET.parse(xml_file).getroot()
    except Exception as e:
        # Accept mislabeled YOLO txt files as a fallback to keep training robust.
        fb_labels, fb_desc, fb_warnings = _parse_yolo_rows_fallback(xml_file)
        if len(fb_labels):
            return fb_labels, fb_desc, fb_warnings
        return np.zeros((0, 5), dtype=np.float32), [], [f"xml parse failed and yolo fallback failed: {e}"]

    for obj in list(root):
        if not isinstance(obj.tag, str) or _local_tag(obj.tag) != "object":
            continue

        class_name = _find_text(obj, "name", recursive=True)
        if not class_name:
            warnings.append("object missing class name")
            continue
        class_id = name_to_id.get(class_name)
        if class_id is None:
            warnings.append(f"unknown class '{class_name}'")
            continue

        bbox = _find_child_by_tag(obj, "bndbox")
        if bbox is None:
            warnings.append(f"object '{class_name}' missing bndbox")
            continue

        xmin = _to_float(_find_text(bbox, "xmin"))
        ymin = _to_float(_find_text(bbox, "ymin"))
        xmax = _to_float(_find_text(bbox, "xmax"))
        ymax = _to_float(_find_text(bbox, "ymax"))
        if xmin is None or ymin is None or xmax is None or ymax is None:
            warnings.append(f"object '{class_name}' has invalid bbox")
            continue

        x1 = max(0.0, min(float(xmin), float(w)))
        y1 = max(0.0, min(float(ymin), float(h)))
        x2 = max(0.0, min(float(xmax), float(w)))
        y2 = max(0.0, min(float(ymax), float(h)))
        if x2 <= x1 or y2 <= y1:
            warnings.append(f"object '{class_name}' has empty bbox")
            continue

        cx = ((x1 + x2) * 0.5) / float(w)
        cy = ((y1 + y2) * 0.5) / float(h)
        bw = (x2 - x1) / float(w)
        bh = (y2 - y1) / float(h)
        if bw <= 0 or bh <= 0:
            warnings.append(f"object '{class_name}' has non-positive bbox size")
            continue

        description = _find_text(obj, "description", recursive=True)
        if not description:
            description = (obj.attrib.get("description", "") or "").strip()

        rows.append([float(class_id), float(cx), float(cy), float(bw), float(bh)])
        descriptions.append(description)

    if not rows:
        return np.zeros((0, 5), dtype=np.float32), [], warnings

    labels = np.asarray(rows, dtype=np.float32)
    return labels, descriptions, warnings


def _verify_image_xml_label(args):
    im_file, xml_file, prefix, name_to_id = args
    nm, nf, ne, nc = 0, 0, 0, 0
    msg = ""

    try:
        with Image.open(im_file) as im:
            im.verify()
        with Image.open(im_file) as im:
            shape_hw = exif_size(im)
            image_format = (im.format or "").lower()

        shape = (shape_hw[1], shape_hw[0])
        assert (shape[0] > 9) and (shape[1] > 9), f"image size {shape} <10 pixels"
        assert image_format in IMG_FORMATS, f"invalid image format {image_format}. {FORMATS_HELP_MSG}"

        if image_format in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING {im_file}: corrupt JPEG restored and saved"

        if os.path.isfile(xml_file):
            nf = 1
            if _looks_like_xml_file(xml_file):
                lb, descriptions, parse_warnings = _parse_xml_rows(xml_file, name_to_id, shape)
            else:
                lb, descriptions, parse_warnings = _parse_yolo_rows_fallback(xml_file)
            if nl := len(lb):
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    i = np.sort(i)
                    lb = lb[i]
                    descriptions = [descriptions[int(j)] for j in i]
                    parse_warnings.append(f"{nl - len(i)} duplicate labels removed")

                # If labels were recovered successfully, suppress legacy fallback-only XML warnings.
                parse_warnings = [
                    w
                    for w in parse_warnings
                    if not (
                        "xml parse failed" in str(w).lower()
                        and "fallback" in str(w).lower()
                    )
                ]
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)

            if parse_warnings:
                detail = "; ".join(parse_warnings[:3])
                suffix = " ..." if len(parse_warnings) > 3 else ""
                msg = f"{prefix}WARNING {im_file}: {detail}{suffix}"
        else:
            nm = 1
            descriptions = []
            lb = np.zeros((0, 5), dtype=np.float32)

        return im_file, lb, shape, descriptions, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, [], nm, nf, ne, nc, msg]


class TextGuidedYOLODataset(YOLODataset):
    """YOLO dataset variant with XML label parsing and single-target object filtering.
    
    For visual grounding: each sample loads (Image, 1 Text Vector, 1 Target BBox).
    Non-target bounding boxes are filtered out so the model only predicts the queried object.
    """

    @staticmethod
    def _is_xml_label_mode(label_files: List[str]) -> bool:
        if not label_files:
            return False
        if not all(str(p).lower().endswith(".xml") for p in label_files):
            return False

        # Verify a small sample of existing files to avoid false XML mode on wrong mappings.
        checked = 0
        for p in label_files:
            if not os.path.isfile(p):
                continue
            checked += 1
            if not _looks_like_xml_file(str(p)):
                return False
            if checked >= 8:
                break
        return True

    @staticmethod
    def _finalize_text_fields(labels, label_files: List[str]):
        for i, lb in enumerate(labels):
            label_file = str(label_files[i]) if i < len(label_files) else ""
            existing_sample_id = str(lb.get("sample_id", "") or "").strip()
            sample_id = existing_sample_id or (Path(label_file).stem if label_file else Path(str(lb.get("im_file", ""))).stem)
            raw_descriptions = lb.get("descriptions")
            if isinstance(raw_descriptions, (list, tuple)):
                descriptions = [str(text).strip() for text in raw_descriptions if str(text).strip()]
            else:
                legacy = str(lb.get("description", "") or "").strip()
                descriptions = [legacy] if legacy else []

            lb["sample_id"] = sample_id
            lb["label_file"] = str(lb.get("label_file", "") or label_file)
            lb["descriptions"] = descriptions
            lb["description"] = descriptions[0] if descriptions else ""
            lb["description_text"] = " ".join(descriptions).strip()
        return labels

    def _build_name_to_id(self) -> Dict[str, int]:
        names = self.data.get("names", {}) if isinstance(self.data, dict) else {}
        mapping: Dict[str, int] = {}

        if isinstance(names, dict):
            for idx, name in names.items():
                label = str(name).strip()
                if not label:
                    continue
                mapping[label] = int(idx)
        elif isinstance(names, (list, tuple)):
            for idx, name in enumerate(names):
                label = str(name).strip()
                if not label:
                    continue
                mapping[label] = int(idx)

        return mapping

    def cache_labels(self, path=Path("./labels.cache")):
        if self.use_keypoints or self.use_segments or self.use_obb:
            raise NotImplementedError("XML label parsing currently supports detect task only.")

        name_to_id = self._build_name_to_id()
        if not name_to_id:
            raise ValueError("Empty class names in data config; cannot map XML class names to class ids.")

        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=_verify_image_xml_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(name_to_id),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, descriptions, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    clean_descriptions = [str(text).strip() for text in descriptions if str(text).strip()]
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": [],
                            "keypoints": None,
                            "normalized": True,
                            "bbox_format": "xywh",
                            "descriptions": clean_descriptions,
                            "description": clean_descriptions[0] if clean_descriptions else "",
                            "description_text": " ".join(clean_descriptions).strip(),
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING No labels found in {path}. {HELP_URL}")

        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        x["label_format"] = "xml"
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        paired_label_files = getattr(self, "paired_label_files", [])
        if paired_label_files and len(paired_label_files) == len(self.im_files):
            self.label_files = paired_label_files
        else:
            self.label_files = img2label_paths(self.im_files)

        if not self._is_xml_label_mode(self.label_files):
            labels = super().get_labels()

            meta_by_pair = {}
            meta_by_label = {}
            img_paths = self.img_path if isinstance(self.img_path, list) else [self.img_path]
            for p in img_paths:
                p = Path(p)
                if p.is_file() and p.suffix.lower() == ".jsonl":
                    import json
                    try:
                        with open(p, "r", encoding="utf-8") as t:
                            for line in t:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    row = json.loads(line)
                                    image_ref = str(row.get("image", "") or row.get("im_file", "")).strip()
                                    label_ref = str(row.get("label", "") or row.get("label_file", "")).strip()
                                    if not image_ref or not label_ref:
                                        continue

                                    image_path = Path(image_ref)
                                    if not image_path.is_absolute():
                                        image_path = (p.parent / image_path).resolve()

                                    label_path = Path(label_ref)
                                    if not label_path.is_absolute():
                                        label_path = (p.parent / label_path).resolve()

                                    descs = row.get("descriptions", [])
                                    if not isinstance(descs, list) or not descs:
                                        desc = str(row.get("description", "") or "").strip()
                                        descs = [desc] if desc else []
                                    descs = [str(d).strip() for d in descs if str(d).strip()]

                                    sample_id = str(row.get("sample_id", "") or row.get("id", "") or "").strip()
                                    if not sample_id:
                                        sample_id = label_path.stem

                                    meta = {
                                        "descriptions": descs,
                                        "sample_id": sample_id,
                                        "label_file": str(label_path),
                                    }
                                    pair_key = (str(image_path), str(label_path))
                                    meta_by_pair[pair_key] = meta
                                    meta_by_label[str(label_path)] = meta
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        LOGGER.warning(f"Failed to parse descriptions from {p}: {e}")

            if meta_by_pair or meta_by_label:
                for i, lb in enumerate(labels):
                    im_file = str(Path(str(lb.get("im_file", ""))).resolve())
                    label_file = ""
                    if i < len(self.label_files):
                        label_file = str(Path(str(self.label_files[i])).resolve())
                    meta = meta_by_pair.get((im_file, label_file)) or meta_by_label.get(label_file)
                    if not meta:
                        continue
                    lb["descriptions"] = list(meta.get("descriptions", []))
                    lb["description"] = lb["descriptions"][0] if lb["descriptions"] else ""
                    lb["sample_id"] = str(meta.get("sample_id", "") or "").strip() or Path(label_file).stem
                    lb["label_file"] = str(meta.get("label_file", "") or label_file)

            return self._finalize_text_fields(labels, getattr(self, "label_files", []))

        # Use xml-specific cache path and marker to avoid stale txt-parser caches.
        cache_path = Path(self.label_files[0]).parent.with_suffix(".xml.cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
            assert cache.get("label_format") == "xml"
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache.get("msgs"):
                legacy_filtered = [
                    m
                    for m in cache["msgs"]
                    if "xml parse failed, used yolo fallback:" not in str(m)
                ]
                if legacy_filtered:
                    LOGGER.info("\n".join(legacy_filtered))

        for key in ("hash", "version", "msgs", "label_format"):
            cache.pop(key, None)

        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING No images found in {cache_path}, training may not work correctly. {HELP_URL}")

        self.im_files = [lb["im_file"] for lb in labels]

        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb.get("segments", []))) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                "WARNING Box and segment counts should be equal, but got "
                f"len(segments) = {len_segments}, len(boxes) = {len_boxes}. "
                "Only boxes will be used and all segments will be removed."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING No labels found in {cache_path}, training may not work correctly. {HELP_URL}")

        return self._finalize_text_fields(labels, self.label_files)

    def __getitem__(self, index: int):
        # Use BaseDataset/YOLODataset default path so one record maps to one target from object-level label files.
        return super().__getitem__(index)
