import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

# Common image suffixes to resolve image path by id.
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _local_tag(tag: str) -> str:
    """Return tag name without XML namespace prefix."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_child_by_tag(node: Optional[ET.Element], tag: str) -> Optional[ET.Element]:
    if node is None:
        return None
    for child in list(node):
        if isinstance(child.tag, str) and _local_tag(child.tag) == tag:
            return child
    return None


def _find_text(node: Optional[ET.Element], tag: str, recursive: bool = False) -> str:
    """Find text from child tag, optionally searching recursively, namespace-safe."""
    if node is None:
        return ""

    elements = node.iter() if recursive else list(node)
    for elem in elements:
        if not isinstance(elem.tag, str):
            continue
        if _local_tag(elem.tag) == tag:
            text = (elem.text or "").strip()
            if text:
                return text
    return ""


def _read_split_ids(split_file: Path) -> List[str]:
    ids: List[str] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def _resolve_images_dir(voc_root: Path, images_dir_name: str) -> Path:
    preferred = voc_root / images_dir_name
    if preferred.is_dir():
        return preferred

    # Compatible with both correct and typo folder names.
    for alt in ("JPGEImages", "JPEGImages"):
        p = voc_root / alt
        if p.is_dir():
            return p

    raise FileNotFoundError(f"Image directory not found under: {voc_root}")


def _resolve_image_path(images_dir: Path, image_id: str, xml_filename: str) -> Optional[Path]:
    candidates: List[Path] = []
    if xml_filename:
        candidates.append(images_dir / xml_filename)

    explicit = images_dir / image_id
    if explicit.suffix:
        candidates.append(explicit)

    for suffix in sorted(VALID_IMAGE_SUFFIXES):
        candidates.append(images_dir / f"{image_id}{suffix}")
        candidates.append(images_dir / f"{image_id}{suffix.upper()}")

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _to_int(text: Optional[str], default: int = 0) -> int:
    if text is None:
        return default
    text = text.strip()
    if not text:
        return default
    return int(float(text))


def _parse_voc_xml(xml_path: Path) -> Dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    xml_filename = _find_text(root, "filename")
    size_node = _find_child_by_tag(root, "size")
    width = _to_int(_find_text(size_node, "width"))
    height = _to_int(_find_text(size_node, "height"))
    objects: List[Dict[str, Any]] = []

    for obj in list(root):
        if not isinstance(obj.tag, str) or _local_tag(obj.tag) != "object":
            continue

        name = _find_text(obj, "name")
        description = _find_text(obj, "description", recursive=True)
        if not description:
            description = (obj.attrib.get("description", "") or "").strip()

        bbox = _find_child_by_tag(obj, "bndbox")
        if not name or bbox is None:
            continue

        xmin = _to_int(_find_text(bbox, "xmin"))
        ymin = _to_int(_find_text(bbox, "ymin"))
        xmax = _to_int(_find_text(bbox, "xmax"))
        ymax = _to_int(_find_text(bbox, "ymax"))
        objects.append(
            {
                "class_name": name,
                "description": description,
                "bbox": [xmin, ymin, xmax, ymax],
            }
        )

    return {
        "filename": xml_filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def _clamp_bbox(bbox: List[int], width: int, height: int) -> Optional[List[int]]:
    if width <= 1 or height <= 1:
        return None
    xmin, ymin, xmax, ymax = bbox
    x1 = max(0, min(xmin, width - 1))
    y1 = max(0, min(ymin, height - 1))
    x2 = max(x1 + 1, min(xmax, width))
    y2 = max(y1 + 1, min(ymax, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _normalize_bbox_xyxy(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> List[float]:
    bw = (x2 - x1) / float(width)
    bh = (y2 - y1) / float(height)
    cx = ((x1 + x2) / 2.0) / float(width)
    cy = ((y1 + y2) / 2.0) / float(height)
    return [cx, cy, bw, bh]


def prepare_voc_object_level_dataset(
    args: Any,
    text_splits: Optional[List[str]] = None,
    device: str = "auto",
    split_files: Optional[Dict[str, str]] = None,
    text_runtime_meta: Optional[Dict[str, Any]] = None,
    text_batch_size: int = 128,
    return_samples: bool = False,
) -> Dict[str, Any]:
    """
        Convert VOC XML dataset to YOLO format with one object per sample.

        Each sample contains one image and a single object in that image:
            - descriptions: List[str] (contains 1 element)
            - source_bboxes: List[[x1, y1, x2, y2]] (contains 1 element)
            - class_ids: List[int] (contains 1 element)
            - class_names: List[str] (contains 1 element)

    Expected VOC layout:
      - {voc_root}/{images_dir_name}
      - {voc_root}/{annotations_dir_name}
      - {voc_root}/train.txt, val.txt, test.txt (image ids without suffix)

    Output layout:
        - {output_root}/labels/{split}/{image_id}_{idx}.txt
      - {output_root}/samples/{split}_samples.jsonl
      - {output_root}/data.yaml
        - {output_root}/*_text_embeddings.pt (optional)
        - {output_root}/embeddings_index.pt (optional)
    """

    voc_root_path = Path(args.voc_root)
    output_root_path = Path(args.prepared_dir)
    annotations_dir = voc_root_path / args.annotations_dir
    images_dir = _resolve_images_dir(voc_root_path, args.images_dir)

    # Ensure deprecated side outputs are not generated/kept.
    for stale_dir in (
        output_root_path / "descriptions",
        output_root_path / "metadata",
        output_root_path / "images",
        output_root_path / "samples",
        output_root_path / "labels",
    ):
        if stale_dir.exists() and stale_dir.is_dir():
            shutil.rmtree(stale_dir)

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    split_files = split_files or {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

    samples_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    sample_files_by_split: Dict[str, str] = {}
    class_to_id: Dict[str, int] = {}
    names: List[str] = []
    errors: List[str] = []

    for split in ("train", "val", "test"):
        split_rows: List[Dict[str, Any]] = []
        samples_file = output_root_path / "samples" / f"{split}_samples.jsonl"
        sample_files_by_split[split] = str(samples_file.relative_to(output_root_path).as_posix())

        split_name = split_files.get(split)
        if not split_name:
            _write_jsonl(samples_file, split_rows)
            continue

        split_txt = voc_root_path / split_name
        if not split_txt.exists():
            _write_jsonl(samples_file, split_rows)
            continue

        image_ids = _read_split_ids(split_txt)
        out_lbl_dir = output_root_path / "labels" / split
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            xml_path = annotations_dir / f"{image_id}.xml"
            if not xml_path.exists():
                errors.append(f"Missing XML: {xml_path}")
                continue

            xml_data = _parse_voc_xml(xml_path)
            image_path = _resolve_image_path(images_dir, image_id, xml_data["filename"])
            if image_path is None:
                errors.append(f"Missing image for id: {image_id}")
                continue

            w = int(xml_data.get("width") or 0)
            h = int(xml_data.get("height") or 0)
            if w <= 1 or h <= 1:
                errors.append(f"Missing or invalid image size in XML: {xml_path}")
                continue

            for obj_idx, obj in enumerate(xml_data["objects"]):
                class_name = obj["class_name"]
                if class_name not in class_to_id:
                    class_to_id[class_name] = len(names)
                    names.append(class_name)
                class_id = class_to_id[class_name]

                clamped = _clamp_bbox(obj["bbox"], w, h)
                if clamped is None:
                    errors.append(f"Invalid bbox in {xml_path} object #{obj_idx}")
                    continue
                x1, y1, x2, y2 = clamped

                label_xywh = _normalize_bbox_xyxy(x1, y1, x2, y2, w, h)
                desc = str(obj.get("description", "") or "").strip()
                
                # Use split-prefixed ID to avoid cross-split collisions for the same image/object index.
                obj_id = f"{split}__{image_id}_{obj_idx:04d}"
                out_lbl = out_lbl_dir / f"{obj_id}.txt"
                
                with open(out_lbl, "w", encoding="utf-8") as f:
                    cx, cy, bw, bh = label_xywh
                    f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                row = {
                    "id": obj_id,
                    "sample_id": obj_id,
                    "split": split,
                    "image": image_path.resolve().as_posix(),
                    "label": out_lbl.resolve().as_posix(),
                    "source_image": str(image_path.relative_to(voc_root_path).as_posix()),
                    "source_xml": str(xml_path.relative_to(voc_root_path).as_posix()),
                    "sample_size": [w, h],
                    "num_objects": 1,
                    "descriptions": [desc],
                    "text": desc,
                    "source_bboxes": [[x1, y1, x2, y2]],
                    "bbox_xyxy_abs": [x1, y1, x2, y2],
                    "bbox_xywh_norm": label_xywh,
                    "class_ids": [int(class_id)],
                    "class_names": [class_name],
                    "description": desc,
                    "source_bbox": [x1, y1, x2, y2],
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "sample_image_mode": "object",
                }
                split_rows.append(row)

        samples_by_split[split] = split_rows
        _write_jsonl(samples_file, split_rows)

    all_samples: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        rows = samples_by_split.get(split, [])
        all_samples.extend(rows)

    data_yaml = output_root_path / "data.yaml"
    data_payload: Dict[str, Any] = {
        "path": str(output_root_path.resolve()),
        "train": sample_files_by_split["train"],
        "val": sample_files_by_split["val"],
        "names": names,
    }
    if split_files.get("test"):
        data_payload["test"] = sample_files_by_split["test"]
    _save_yaml(data_yaml, data_payload)

    text_result: Dict[str, Any] = {}
    selected_splits = [str(s).strip() for s in (text_splits or []) if str(s).strip()]
    if not selected_splits:
        selected_splits = [s for s in ("train", "val") if samples_by_split.get(s)]

    if selected_splits:
        from dataset.precompute_text_embeddings import build_openclip_text_embeddings

        text_model_name = str(getattr(args, "text_model_name", "ViT-L-14") or "ViT-L-14")
        text_pretrained = str(getattr(args, "text_pretrained", "openai") or "openai")
        text_precision = str(getattr(args, "text_precision", "auto") or "auto")
        text_overwrite = bool(getattr(args, "text_overwrite", False))
        text_empty_placeholder = str(getattr(args, "text_empty_placeholder", "[NO_DESCRIPTION]") or "[NO_DESCRIPTION]")
        text_max_samples = int(getattr(args, "text_max_samples", 0) or 0)
        text_no_save_texts = bool(getattr(args, "text_no_save_texts", False))
        text_no_save_index = bool(getattr(args, "text_no_save_index", False))

        text_result = build_openclip_text_embeddings(
            prepared_dir=str(output_root_path),
            splits=selected_splits,
            output_dir=str(output_root_path),
            model_name=text_model_name,
            pretrained=text_pretrained,
            device=device,
            precision=text_precision,
            batch_size=int(text_batch_size) if int(text_batch_size) > 0 else 128,
            overwrite=text_overwrite,
            empty_placeholder=text_empty_placeholder,
            max_samples=text_max_samples,
            save_texts=not text_no_save_texts,
            save_index=not text_no_save_index,
            runtime_meta=dict(text_runtime_meta or {}),
            split_samples=samples_by_split,
        )

    result: Dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "object_image_mode": "object",
        "sample_granularity": "object",
        "names": names,
        "num_classes": len(names),
        "samples_per_split": {
            "train": len(samples_by_split.get("train", [])),
            "val": len(samples_by_split.get("val", [])),
            "test": len(samples_by_split.get("test", [])),
        },
        "num_samples": len(all_samples),
        "errors": errors,
        "text_embeddings": text_result,
    }
    if return_samples:
        result["samples"] = all_samples
    return result


def prepare_voc_image_level_dataset(
    args: Any,
    text_splits: Optional[List[str]] = None,
    device: str = "auto",
    split_files: Optional[Dict[str, str]] = None,
    text_runtime_meta: Optional[Dict[str, Any]] = None,
    text_batch_size: int = 128,
    return_samples: bool = False,
) -> Dict[str, Any]:
    """
    Convert VOC XML dataset to YOLO format with one image per sample.

    Each sample contains one image and all objects in that image:
        - descriptions: List[str] (len == num_objects)
        - source_bboxes: List[[x1, y1, x2, y2]] (len == num_objects)
        - class_ids: List[int] (len == num_objects)
        - class_names: List[str] (len == num_objects)
    """

    voc_root_path = Path(args.voc_root)
    output_root_path = Path(args.prepared_dir)
    annotations_dir = voc_root_path / args.annotations_dir
    images_dir = _resolve_images_dir(voc_root_path, args.images_dir)

    for stale_dir in (
        output_root_path / "descriptions",
        output_root_path / "metadata",
        output_root_path / "images",
        output_root_path / "samples",
        output_root_path / "labels",
    ):
        if stale_dir.exists() and stale_dir.is_dir():
            shutil.rmtree(stale_dir)

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    split_files = split_files or {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

    samples_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    sample_files_by_split: Dict[str, str] = {}
    class_to_id: Dict[str, int] = {}
    names: List[str] = []
    errors: List[str] = []

    for split in ("train", "val", "test"):
        split_rows: List[Dict[str, Any]] = []
        samples_file = output_root_path / "samples" / f"{split}_samples.jsonl"
        sample_files_by_split[split] = str(samples_file.relative_to(output_root_path).as_posix())

        split_name = split_files.get(split)
        if not split_name:
            _write_jsonl(samples_file, split_rows)
            continue

        split_txt = voc_root_path / split_name
        if not split_txt.exists():
            _write_jsonl(samples_file, split_rows)
            continue

        image_ids = _read_split_ids(split_txt)
        out_lbl_dir = output_root_path / "labels" / split
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            xml_path = annotations_dir / f"{image_id}.xml"
            if not xml_path.exists():
                errors.append(f"Missing XML: {xml_path}")
                continue

            xml_data = _parse_voc_xml(xml_path)
            image_path = _resolve_image_path(images_dir, image_id, xml_data["filename"])
            if image_path is None:
                errors.append(f"Missing image for id: {image_id}")
                continue

            w = int(xml_data.get("width") or 0)
            h = int(xml_data.get("height") or 0)
            if w <= 1 or h <= 1:
                errors.append(f"Missing or invalid image size in XML: {xml_path}")
                continue

            sample_id = f"{split}__{image_id}"
            out_lbl = out_lbl_dir / f"{sample_id}.txt"

            descriptions: List[str] = []
            source_bboxes: List[List[int]] = []
            class_ids: List[int] = []
            class_names: List[str] = []
            yolo_rows: List[List[float]] = []

            for obj_idx, obj in enumerate(xml_data["objects"]):
                class_name = obj["class_name"]
                if class_name not in class_to_id:
                    class_to_id[class_name] = len(names)
                    names.append(class_name)
                class_id = class_to_id[class_name]

                clamped = _clamp_bbox(obj["bbox"], w, h)
                if clamped is None:
                    errors.append(f"Invalid bbox in {xml_path} object #{obj_idx}")
                    continue
                x1, y1, x2, y2 = clamped

                label_xywh = _normalize_bbox_xyxy(x1, y1, x2, y2, w, h)
                desc = str(obj.get("description", "") or "").strip()

                yolo_rows.append([float(class_id), label_xywh[0], label_xywh[1], label_xywh[2], label_xywh[3]])
                descriptions.append(desc)
                source_bboxes.append([x1, y1, x2, y2])
                class_ids.append(int(class_id))
                class_names.append(class_name)

            if not yolo_rows:
                continue

            with open(out_lbl, "w", encoding="utf-8") as f:
                for class_id, cx, cy, bw, bh in yolo_rows:
                    f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            first_desc = descriptions[0] if descriptions else ""
            first_box = source_bboxes[0] if source_bboxes else []
            first_class_id = class_ids[0] if class_ids else -1
            first_class_name = class_names[0] if class_names else ""

            row = {
                "id": sample_id,
                "sample_id": sample_id,
                "split": split,
                "image": image_path.resolve().as_posix(),
                "label": out_lbl.resolve().as_posix(),
                "source_image": str(image_path.relative_to(voc_root_path).as_posix()),
                "source_xml": str(xml_path.relative_to(voc_root_path).as_posix()),
                "sample_size": [w, h],
                "num_objects": len(class_ids),
                "descriptions": descriptions,
                "text": " ".join([d for d in descriptions if d]).strip(),
                "source_bboxes": source_bboxes,
                "class_ids": class_ids,
                "class_names": class_names,
                "description": first_desc,
                "source_bbox": first_box,
                "class_id": int(first_class_id),
                "class_name": first_class_name,
                "sample_image_mode": "image",
            }
            split_rows.append(row)

        samples_by_split[split] = split_rows
        _write_jsonl(samples_file, split_rows)

    all_samples: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        rows = samples_by_split.get(split, [])
        all_samples.extend(rows)

    data_yaml = output_root_path / "data.yaml"
    data_payload: Dict[str, Any] = {
        "path": str(output_root_path.resolve()),
        "train": sample_files_by_split["train"],
        "val": sample_files_by_split["val"],
        "names": names,
    }
    if split_files.get("test"):
        data_payload["test"] = sample_files_by_split["test"]
    _save_yaml(data_yaml, data_payload)

    text_result: Dict[str, Any] = {}
    selected_splits = [str(s).strip() for s in (text_splits or []) if str(s).strip()]
    if not selected_splits:
        selected_splits = [s for s in ("train", "val") if samples_by_split.get(s)]

    if selected_splits:
        from dataset.precompute_text_embeddings import build_openclip_text_embeddings

        text_model_name = str(getattr(args, "text_model_name", "ViT-L-14") or "ViT-L-14")
        text_pretrained = str(getattr(args, "text_pretrained", "openai") or "openai")
        text_precision = str(getattr(args, "text_precision", "auto") or "auto")
        text_overwrite = bool(getattr(args, "text_overwrite", False))
        text_empty_placeholder = str(getattr(args, "text_empty_placeholder", "[NO_DESCRIPTION]") or "[NO_DESCRIPTION]")
        text_max_samples = int(getattr(args, "text_max_samples", 0) or 0)
        text_no_save_texts = bool(getattr(args, "text_no_save_texts", False))
        text_no_save_index = bool(getattr(args, "text_no_save_index", False))

        text_result = build_openclip_text_embeddings(
            prepared_dir=str(output_root_path),
            splits=selected_splits,
            output_dir=str(output_root_path),
            model_name=text_model_name,
            pretrained=text_pretrained,
            device=device,
            precision=text_precision,
            batch_size=int(text_batch_size) if int(text_batch_size) > 0 else 128,
            overwrite=text_overwrite,
            empty_placeholder=text_empty_placeholder,
            max_samples=text_max_samples,
            save_texts=not text_no_save_texts,
            save_index=not text_no_save_index,
            runtime_meta=dict(text_runtime_meta or {}),
            split_samples=samples_by_split,
        )

    result: Dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "object_image_mode": "image",
        "sample_granularity": "image",
        "names": names,
        "num_classes": len(names),
        "samples_per_split": {
            "train": len(samples_by_split.get("train", [])),
            "val": len(samples_by_split.get("val", [])),
            "test": len(samples_by_split.get("test", [])),
        },
        "num_samples": len(all_samples),
        "errors": errors,
        "text_embeddings": text_result,
    }
    if return_samples:
        result["samples"] = all_samples
    return result


def _read_yolo_label_file(label_path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    if not label_path.exists():
        return rows

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return rows


def _is_valid_yolo_label_rows(rows: List[List[float]]) -> bool:
    if not rows:
        return False
    for class_id, cx, cy, bw, bh in rows:
        if class_id < 0:
            return False
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            return False
    return True


def validate_prepared_yolo_dataset(prepared_root: str, required_splits: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate whether the converted dataset can be consumed directly by YOLO detection training."""
    root = Path(prepared_root)
    required_splits = required_splits or ["train", "val"]

    issues: List[str] = []
    stats: Dict[str, Dict[str, int]] = {}

    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return {
            "valid": False,
            "issues": [f"Missing data.yaml: {data_yaml}"],
            "stats": {},
        }

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for split in required_splits:
        if not data.get(split):
            issues.append(f"Missing '{split}' entry in data.yaml")

    for split in ("train", "val", "test"):
        rel = data.get(split)
        if not rel:
            continue

        split_path = Path(str(rel))
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()

        missing_labels = 0
        invalid_labels = 0
        valid_labels = 0
        missing_images = 0

        if split_path.is_file() and split_path.suffix.lower() == ".jsonl":
            samples = load_object_descriptions(str(split_path))

            for row in samples:
                image_ref = str(row.get("image", "") or row.get("im_file", "")).strip()
                label_ref = str(row.get("label", "") or row.get("label_file", "")).strip()

                if not image_ref:
                    missing_images += 1
                    continue
                if not label_ref:
                    missing_labels += 1
                    continue

                image_path = Path(image_ref)
                if not image_path.is_absolute():
                    image_path = (split_path.parent / image_path).resolve()

                label_path = Path(label_ref)
                if not label_path.is_absolute():
                    label_path = (split_path.parent / label_path).resolve()

                if not image_path.exists():
                    missing_images += 1
                if not label_path.exists():
                    missing_labels += 1
                    continue

                label_rows = _read_yolo_label_file(label_path)
                if _is_valid_yolo_label_rows(label_rows):
                    valid_labels += 1
                else:
                    invalid_labels += 1

            stats[split] = {
                "images": len(samples),
                "valid_labels": valid_labels,
                "missing_labels": missing_labels,
                "invalid_labels": invalid_labels,
                "missing_images": missing_images,
            }

            if missing_images > 0:
                issues.append(f"Split '{split}' has {missing_images} samples with missing source images")
            if missing_labels > 0:
                issues.append(f"Split '{split}' has {missing_labels} samples without label files")
            if invalid_labels > 0:
                issues.append(f"Split '{split}' has {invalid_labels} invalid label files")
            continue

        image_dir = split_path
        label_dir = root / "labels" / split

        if not image_dir.is_dir():
            issues.append(f"Missing image dir for split '{split}': {image_dir}")
            continue
        if not label_dir.is_dir():
            issues.append(f"Missing label dir for split '{split}': {label_dir}")
            continue

        image_files = [p for p in image_dir.rglob("*") if p.suffix.lower() in VALID_IMAGE_SUFFIXES]

        for img_path in image_files:
            lbl = label_dir / f"{img_path.stem}.txt"
            if not lbl.exists():
                missing_labels += 1
                continue

            label_rows = _read_yolo_label_file(lbl)
            if _is_valid_yolo_label_rows(label_rows):
                valid_labels += 1
            else:
                invalid_labels += 1

        stats[split] = {
            "images": len(image_files),
            "valid_labels": valid_labels,
            "missing_labels": missing_labels,
            "invalid_labels": invalid_labels,
        }

        if missing_labels > 0:
            issues.append(f"Split '{split}' has {missing_labels} images without label files")
        if invalid_labels > 0:
            issues.append(f"Split '{split}' has {invalid_labels} invalid label files")

    if not data.get("names"):
        issues.append("Missing 'names' in data.yaml")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": stats,
        "data_yaml": str(data_yaml),
    }


def load_object_descriptions(description_file: str) -> List[Dict[str, Any]]:
    """Load sample rows from JSONL."""
    rows: List[Dict[str, Any]] = []
    with open(description_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sample_rows(samples_file: str) -> List[Dict[str, Any]]:
    """Alias of load_object_descriptions for semantic clarity."""
    return load_object_descriptions(samples_file)


def _build_split_samples_without_metadata(root: Path, split: str) -> List[Dict[str, Any]]:
    samples_jsonl = root / "samples" / f"{split}_samples.jsonl"
    if samples_jsonl.exists():
        rows = load_object_descriptions(str(samples_jsonl))
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            sample = dict(row)
            sample.setdefault("id", f"{split}__row{i:08d}")
            sample.setdefault("split", split)
            sample.setdefault("description", "")

            descriptions = sample.get("descriptions")
            if not isinstance(descriptions, list):
                legacy = str(sample.get("description", "") or "").strip()
                descriptions = [legacy] if legacy else []
            sample["descriptions"] = [str(t) for t in descriptions if str(t).strip()]

            class_ids = sample.get("class_ids")
            if not isinstance(class_ids, list):
                try:
                    legacy_class = int(sample.get("class_id", -1))
                except Exception:
                    legacy_class = -1
                class_ids = [legacy_class] if legacy_class >= 0 else []
            sample["class_ids"] = [int(c) for c in class_ids if int(c) >= 0]

            class_names = sample.get("class_names")
            if not isinstance(class_names, list):
                legacy_name = str(sample.get("class_name", "") or "").strip()
                class_names = [legacy_name] if legacy_name else []
            sample["class_names"] = [str(x) for x in class_names if str(x).strip()]

            source_bboxes = sample.get("source_bboxes")
            if not isinstance(source_bboxes, list):
                legacy_box = sample.get("source_bbox")
                source_bboxes = [legacy_box] if isinstance(legacy_box, (list, tuple)) and len(legacy_box) == 4 else []
            normalized_boxes: List[List[int]] = []
            for box in source_bboxes:
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    normalized_boxes.append([int(float(v)) for v in box])
                except Exception:
                    continue
            sample["source_bboxes"] = normalized_boxes
            sample.setdefault("num_objects", len(sample["class_ids"]))

            if "image" not in sample and "im_file" in sample:
                sample["image"] = sample.get("im_file")
            if "label" not in sample and "label_file" in sample:
                sample["label"] = sample.get("label_file")
            if sample.get("image") and sample.get("label"):
                samples.append(sample)
        return samples

    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split folders: images={image_dir}, labels={label_dir}")

    image_files = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in VALID_IMAGE_SUFFIXES])
    samples: List[Dict[str, Any]] = []
    for img_path in image_files:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        label_rows = _read_yolo_label_file(lbl_path)
        class_ids = [int(row[0]) for row in label_rows if int(row[0]) >= 0]

        samples.append(
            {
                "id": img_path.stem,
                "split": split,
                "description": "",
                "descriptions": [],
                "image": str(img_path.relative_to(root).as_posix()),
                "label": str(lbl_path.relative_to(root).as_posix()),
                "source_bboxes": [],
                "class_ids": class_ids,
                "class_names": [],
                "num_objects": len(class_ids),
                "source_bbox": [],
                "class_id": class_ids[0] if class_ids else -1,
                "class_name": "",
            }
        )
    return samples


def _load_description_texts_from_embeddings(root: Path, split: str) -> Dict[str, str]:
    emb_path = root / f"{split}_text_embeddings.pt"
    if not emb_path.exists():
        return {}

    try:
        import torch
    except ImportError:
        return {}

    payload = torch.load(emb_path, map_location="cpu")
    ids = payload.get("ids", [])
    source_texts = payload.get("source_texts", payload.get("texts", []))
    phrases = payload.get("phrases", [])

    text_items = source_texts
    if not isinstance(text_items, (list, tuple)) or len(text_items) != len(ids):
        text_items = phrases
    if not isinstance(text_items, (list, tuple)) or len(text_items) != len(ids):
        return {}

    text_map: Dict[str, str] = {}
    for sample_id, text in zip(ids, text_items):
        if isinstance(text, (list, tuple)):
            merged = " ".join([str(t).strip() for t in text if str(t).strip()])
        else:
            merged = str(text)
        text_map[str(sample_id)] = merged
    return text_map


class ObjectLevelDescriptionDataset:
    """
    Lightweight image-level dataset.

    __getitem__ reads all texts for each image-level sample.
    """

    def __init__(self, prepared_root: str, split: str = "train", load_image: bool = False) -> None:
        self.root = Path(prepared_root)
        self.split = split
        self.load_image = load_image
        self.samples_path = self.root / "samples" / f"{split}_samples.jsonl"
        self.metadata_path = self.root / "metadata" / f"{split}_objects.jsonl"
        if self.samples_path.exists():
            self.samples = load_object_descriptions(str(self.samples_path))
        elif self.metadata_path.exists():
            self.samples = load_object_descriptions(str(self.metadata_path))
        else:
            self.samples = _build_split_samples_without_metadata(self.root, split)
            text_map = _load_description_texts_from_embeddings(self.root, split)
            if text_map:
                for sample in self.samples:
                    sid = sample.get("id", "")
                    merged_text = text_map.get(sid, "")
                    if merged_text:
                        sample["description"] = merged_text
                        sample["descriptions"] = [merged_text]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = dict(self.samples[index])

        image_ref = Path(str(sample.get("image", "")))
        label_ref = Path(str(sample.get("label", "")))
        image_path = image_ref if image_ref.is_absolute() else (self.root / image_ref)
        label_path = label_ref if label_ref.is_absolute() else (self.root / label_ref)

        descriptions = sample.get("descriptions")
        if not isinstance(descriptions, list):
            legacy = str(sample.get("description", "") or "").strip()
            descriptions = [legacy] if legacy else []
        descriptions = [str(text) for text in descriptions if str(text).strip()]
        description_text = " ".join(descriptions).strip()

        source_bboxes = sample.get("source_bboxes")
        if not isinstance(source_bboxes, list):
            legacy_box = sample.get("source_bbox")
            source_bboxes = [legacy_box] if isinstance(legacy_box, (list, tuple)) and len(legacy_box) == 4 else []

        class_ids = sample.get("class_ids")
        if not isinstance(class_ids, list):
            try:
                legacy_class = int(sample.get("class_id", -1))
            except Exception:
                legacy_class = -1
            class_ids = [legacy_class] if legacy_class >= 0 else []

        sample["image_path"] = str(image_path)
        sample["label_path"] = str(label_path)
        sample["descriptions"] = descriptions
        sample["description_text"] = description_text
        sample["source_bboxes"] = source_bboxes
        sample["class_ids"] = [int(c) for c in class_ids if int(c) >= 0]
        sample["yolo_labels"] = _read_yolo_label_file(label_path)

        if self.load_image:
            sample["image_array"] = cv2.imread(str(image_path))

        return sample


ImageLevelDescriptionDataset = ObjectLevelDescriptionDataset
