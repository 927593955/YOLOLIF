import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re
import torch

from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder, sanitize_text


PHRASE_TYPE_ORDER: Tuple[str, ...] = ("NP", "PP", "ADJP")
DEFAULT_PHRASE_TYPE_WEIGHTS: Dict[str, float] = {
    "NP": 1.0,
    "PP": 1.2,
    "ADJP": 0.8,
    "FALLBACK": 1.0,
}


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL rows as dictionaries."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_splits(splits_arg: str) -> List[str]:
    splits = [s.strip() for s in (splits_arg or "").split(",") if s.strip()]
    if not splits:
        raise ValueError("At least one split is required, e.g. --splits train,val")
    return splits


def default_output_dir(prepared_root: Path, model_name: str, pretrained: str) -> Path:
    safe_model = model_name.lower().replace("/", "-").replace(" ", "")
    safe_pretrained = pretrained.lower().replace("/", "-").replace(" ", "")
    return prepared_root / "embeddings" / f"openclip_{safe_model}_{safe_pretrained}"


def _parse_phrase_types(phrase_types: Sequence[str] | str) -> Tuple[str, ...]:
    if isinstance(phrase_types, str):
        parts = [x.strip().upper() for x in phrase_types.split(",") if x.strip()]
    else:
        parts = [str(x).strip().upper() for x in phrase_types if str(x).strip()]

    valid = set(PHRASE_TYPE_ORDER)
    dedup: List[str] = []
    for p in parts:
        if p not in valid:
            continue
        if p not in dedup:
            dedup.append(p)

    if not dedup:
        dedup = list(PHRASE_TYPE_ORDER)
    return tuple(dedup)


def _parse_phrase_weight_string(raw: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in str(raw or "").split(","):
        pair = item.strip()
        if not pair or ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key:
            continue
        try:
            weight = float(value)
        except ValueError:
            continue
        if weight > 0:
            weights[key] = weight
    return weights


def _resolve_phrase_type_weights(
    phrase_types: Iterable[str],
    phrase_type_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    resolved = dict(DEFAULT_PHRASE_TYPE_WEIGHTS)
    for key, value in (phrase_type_weights or {}).items():
        k = str(key).strip().upper()
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0:
            resolved[k] = v

    for p in phrase_types:
        if p not in resolved or float(resolved[p]) <= 0.0:
            resolved[p] = 1.0
    if float(resolved.get("FALLBACK", 0.0)) <= 0.0:
        resolved["FALLBACK"] = 1.0
    return resolved


def _load_spacy_pipeline(model_name: str):
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is required for phrase-level preprocessing. Install with: pip install spacy"
        ) from e

    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            "spaCy model is not available. "
            f"Please run: python -m spacy download {model_name}"
        ) from e
    return nlp

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_phrase_candidates(doc, enabled_types: Set[str]) -> List[Tuple[str, str]]:
    phrases: List[Tuple[str, str]] = []

    # -------- NP --------
    if "NP" in enabled_types:
        try:
            for chunk in doc.noun_chunks:
                text = normalize(sanitize_text(chunk.text, empty_placeholder=""))
                if text:
                    phrases.append((text, "NP"))
        except Exception:
            pass

    # -------- PP --------
    if "PP" in enabled_types:
        for token in doc:
            if token.dep_ != "prep":
                continue
            subtree = list(token.subtree)
            if not subtree:
                continue
            span = doc[subtree[0].i : subtree[-1].i + 1]
            text = normalize(sanitize_text(span.text, empty_placeholder=""))
            if text:
                phrases.append((text, "PP"))

    # -------- ADJP --------
    if "ADJP" in enabled_types:
        allowed_dep = {"amod", "acomp", "attr", "oprd"}
        for token in doc:
            if token.pos_ != "ADJ" or token.dep_ not in allowed_dep:
                continue
            subtree = list(token.subtree)
            span = doc[subtree[0].i : subtree[-1].i + 1] if subtree else doc[token.i : token.i + 1]
            text = normalize(sanitize_text(span.text, empty_placeholder=""))
            if text:
                phrases.append((text, "ADJP"))

    # -------- 去重（完全重复）--------
    phrases = list(set(phrases))

    # -------- 去冗余（核心）--------
    final_phrases: List[Tuple[str, str]] = []

    # 类型优先级：PP > NP > ADJP
    type_priority = {"PP": 3, "NP": 2, "ADJP": 1}

    for i, (p_text, p_type) in enumerate(phrases):
        keep = True
        for j, (q_text, q_type) in enumerate(phrases):
            if i == j:
                continue

            # 如果 p 被 q 包含
            if p_text in q_text:
                # 且 q 更重要（更长 或 类型优先级更高）
                if (len(q_text) > len(p_text)) or (
                    type_priority.get(q_type, 0) > type_priority.get(p_type, 0)
                ):
                    keep = False
                    break

        if keep:
            final_phrases.append((p_text, p_type))

    return final_phrases

'''
def _extract_phrase_candidates(doc, enabled_types: Set[str]) -> List[Tuple[str, str]]:
    phrases: List[Tuple[str, str]] = []

    if "NP" in enabled_types:
        try:
            for chunk in doc.noun_chunks:
                text = sanitize_text(chunk.text, empty_placeholder="")
                if text:
                    phrases.append((text, "NP"))
        except Exception:
            pass

    if "PP" in enabled_types:
        for token in doc:
            if token.dep_ != "prep":
                continue
            subtree = list(token.subtree)
            if not subtree:
                continue
            span = doc[subtree[0].i : subtree[-1].i + 1]
            text = sanitize_text(span.text, empty_placeholder="")
            if text:
                phrases.append((text, "PP"))

    if "ADJP" in enabled_types:
        allowed_dep = {"amod", "acomp", "attr", "oprd", "ROOT"}
        for token in doc:
            if token.pos_ != "ADJ" or token.dep_ not in allowed_dep:
                continue
            subtree = list(token.subtree)
            span = doc[subtree[0].i : subtree[-1].i + 1] if subtree else doc[token.i : token.i + 1]
            text = sanitize_text(span.text, empty_placeholder="")
            if text:
                phrases.append((text, "ADJP"))

    return phrases
'''

def _extract_phrases_for_sample(
    nlp,
    texts: Sequence[str],
    enabled_types: Tuple[str, ...],
    max_phrases_per_sample: int,
    empty_placeholder: str,
) -> Tuple[List[str], List[str], bool]:
    phrase_items: List[Tuple[str, str]] = []
    type_set = set(enabled_types)
    clean_texts = [sanitize_text(t, empty_placeholder="") for t in texts]
    clean_texts = [t for t in clean_texts if t]

    for text in clean_texts:
        doc = nlp(text)
        phrase_items.extend(_extract_phrase_candidates(doc, type_set))

    dedup_text: List[str] = []
    dedup_type: List[str] = []
    seen: Set[str] = set()
    for phrase_text, phrase_type in phrase_items:
        key = phrase_text.casefold()
        if key in seen:
            continue
        seen.add(key)
        dedup_text.append(phrase_text)
        dedup_type.append(phrase_type)
        if max_phrases_per_sample > 0 and len(dedup_text) >= max_phrases_per_sample:
            break

    used_fallback = False
    if not dedup_text:
        fallback_text = sanitize_text(" ".join(clean_texts), empty_placeholder="")
        if not fallback_text:
            fallback_text = empty_placeholder
        dedup_text = [fallback_text]
        dedup_type = ["FALLBACK"]
        used_fallback = True

    return dedup_text, dedup_type, used_fallback


def _read_description_texts(root: Path, sample: Dict[str, Any], empty_placeholder: str) -> Tuple[List[str], str, bool]:
    sources: List[str] = []

    descriptions = sample.get("descriptions")
    if isinstance(descriptions, (list, tuple)):
        desc_texts = [sanitize_text(text, empty_placeholder="") for text in descriptions]
        desc_texts = [text for text in desc_texts if text]
        if desc_texts:
            return desc_texts, "descriptions_field", False
        sources.append("descriptions_field_empty")
    else:
        sources.append("descriptions_field_missing")

    desc_rel = str(sample.get("description_file", "") or "").strip()
    if desc_rel:
        desc_path = root / desc_rel
        if desc_path.exists():
            file_text = desc_path.read_text(encoding="utf-8")
            cleaned_file_text = sanitize_text(file_text, empty_placeholder="")
            if cleaned_file_text:
                return [cleaned_file_text], "description_file", False
            sources.append("description_file_empty")
        else:
            sources.append("description_file_missing")

    single_text = sanitize_text(sample.get("description", ""), empty_placeholder="")
    if single_text:
        return [single_text], "description_field", False
    sources.append("description_field_empty")

    class_names = sample.get("class_names")
    if isinstance(class_names, (list, tuple)):
        class_texts = [sanitize_text(name, empty_placeholder="") for name in class_names]
        class_texts = [text for text in class_texts if text]
        if class_texts:
            return class_texts, "class_names_fallback", True

    class_name = sanitize_text(sample.get("class_name", ""), empty_placeholder="")
    if class_name:
        return [class_name], "class_name_fallback", True

    fallback_source = "+".join(sources) if sources else "placeholder_fallback"
    return [empty_placeholder], f"{fallback_source}+placeholder_fallback", True


def _read_object_texts(sample: Dict[str, Any], empty_placeholder: str) -> Tuple[List[str], bool]:
    """Build object-aligned texts where index k maps to object k."""
    descriptions_raw = sample.get("descriptions")
    class_names_raw = sample.get("class_names")
    num_objects_raw = sample.get("num_objects", 0)

    descriptions = []
    if isinstance(descriptions_raw, (list, tuple)):
        descriptions = [sanitize_text(x, empty_placeholder="") for x in descriptions_raw]

    class_names = []
    if isinstance(class_names_raw, (list, tuple)):
        class_names = [sanitize_text(x, empty_placeholder="") for x in class_names_raw]

    try:
        num_objects = int(num_objects_raw)
    except Exception:
        num_objects = 0
    if num_objects <= 0:
        num_objects = max(len(descriptions), len(class_names))
    num_objects = max(1, num_objects)

    used_fallback = False
    object_texts: List[str] = []
    for obj_idx in range(num_objects):
        desc = descriptions[obj_idx] if obj_idx < len(descriptions) else ""
        if desc:
            object_texts.append(desc)
            continue

        cname = class_names[obj_idx] if obj_idx < len(class_names) else ""
        if cname:
            object_texts.append(cname)
            used_fallback = True
            continue

        object_texts.append(empty_placeholder)
        used_fallback = True

    return object_texts, used_fallback


def _load_split_samples(
    prepared_root: Path,
    split: str,
    split_samples: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    if split_samples is not None:
        rows = split_samples.get(split, [])
        return [dict(row) for row in rows]

    samples_path = prepared_root / "samples" / f"{split}_samples.jsonl"
    if samples_path.exists():
        return _load_jsonl_rows(samples_path)

    metadata_path = prepared_root / "metadata" / f"{split}_objects.jsonl"
    if not metadata_path.exists():
        return []
    return _load_jsonl_rows(metadata_path)


def build_openclip_text_embeddings(
    prepared_dir: str,
    splits: Sequence[str],
    output_dir: str = "",
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "auto",
    precision: str = "auto",
    batch_size: int = 256,
    overwrite: bool = False,
    empty_placeholder: str = "[NO_DESCRIPTION]",
    max_samples: int = 0,
    save_texts: bool = True,
    save_index: bool = True,
    runtime_meta: Optional[Dict[str, Any]] = None,
    split_samples: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    spacy_model: str = "en_core_web_sm",
    phrase_types: Sequence[str] | str = ("NP", "PP", "ADJP"),
    phrase_type_weights: Optional[Dict[str, float]] = None,
    max_phrases_per_sample: int = 24,
) -> Dict[str, Any]:
    root = Path(prepared_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Prepared dataset dir not found: {root}")

    if split_samples is None:
        metadata_dir = root / "metadata"
        samples_dir = root / "samples"
        if not metadata_dir.exists() and not samples_dir.exists():
            raise FileNotFoundError(
                f"Neither metadata dir nor samples dir found: metadata={metadata_dir}, samples={samples_dir}"
            )

    split_list = [str(s).strip() for s in splits if str(s).strip()]
    if not split_list:
        raise ValueError("No valid splits provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if int(max_phrases_per_sample) <= 0:
        raise ValueError("max_phrases_per_sample must be > 0")

    runtime_meta = dict(runtime_meta or {})
    enabled_phrase_types = _parse_phrase_types(phrase_types)
    phrase_weights_by_type = _resolve_phrase_type_weights(enabled_phrase_types, phrase_type_weights)
    nlp = _load_spacy_pipeline(str(spacy_model))

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(root, model_name, pretrained)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = OpenCLIPTextEncoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
        normalize=True,
    )

    duplicate_ids: List[str] = []
    index_map: Dict[str, Dict[str, Any]] = {}
    split_stats: Dict[str, Dict[str, Any]] = {}

    for split in split_list:
        output_file = out_dir / f"{split}_text_embeddings.pt"

        if output_file.exists() and not overwrite:
            payload = torch.load(output_file, map_location="cpu")
            ids_existing = payload.get("ids", [])
            offsets_existing = payload.get("sample_offsets", [])
            if isinstance(offsets_existing, (list, tuple)) and len(offsets_existing) == len(ids_existing) + 1:
                num_texts_existing = int(offsets_existing[-1])
            else:
                embeddings_existing = payload.get("embeddings")
                if isinstance(embeddings_existing, torch.Tensor) and embeddings_existing.ndim == 2:
                    num_texts_existing = int(embeddings_existing.shape[0])
                else:
                    num_texts_existing = len(ids_existing)

            for idx, sample_id in enumerate(ids_existing):
                if sample_id in index_map:
                    duplicate_ids.append(sample_id)
                index_map[sample_id] = {"split": split, "index": idx}

            split_stats[split] = {
                "status": "skipped_existing",
                "output": str(output_file),
                "num_samples": len(ids_existing),
                "num_phrases": num_texts_existing,
                "num_empty_fallback": int(payload.get("preprocess_meta", {}).get("num_empty_fallback", 0)),
                "num_phrase_fallback": int(payload.get("preprocess_meta", {}).get("num_phrase_fallback", 0)),
            }
            continue

        samples = _load_split_samples(root, split, split_samples=split_samples)
        if max_samples > 0:
            samples = samples[:max_samples]

        ids: List[str] = []
        source_texts: List[List[str]] = []
        phrase_texts_by_sample: List[List[str]] = []
        flat_phrases: List[str] = []
        flat_phrase_types: List[str] = []
        flat_phrase_weights: List[float] = []
        flat_token_target_indices: List[int] = []
        sample_offsets: List[int] = [0]
        text_sources: List[str] = []
        num_empty_fallback = 0
        num_phrase_fallback = 0

        for i, sample in enumerate(samples):
            sample_id = str(
                sample.get("sample_id", "")
                or sample.get("id", "")
                or sample.get("object_id", "")
                or ""
            ).strip()
            if not sample_id:
                sample_id = f"{split}__row{i:08d}"

            phrases, used_empty_fallback = _read_object_texts(sample, empty_placeholder)
            phrase_types_for_sample = ["OBJECT" for _ in phrases]
            used_phrase_fallback = False
            source = "object_descriptions_1to1"
            ids.append(sample_id)
            source_texts.append(list(phrases))
            phrase_texts_by_sample.append(phrases)
            text_sources.append(source)
            num_empty_fallback += int(used_empty_fallback)
            num_phrase_fallback += int(used_phrase_fallback)
            flat_phrases.extend(phrases)
            flat_phrase_types.extend(phrase_types_for_sample)
            flat_phrase_weights.extend([1.0 for _ in phrase_types_for_sample])
            flat_token_target_indices.extend(list(range(len(phrases))))
            sample_offsets.append(len(flat_phrases))

        if not ids:
            split_stats[split] = {
                "status": "empty_split",
                "output": str(output_file),
                "num_samples": 0,
                "num_phrases": 0,
                "num_empty_fallback": 0,
            }
            continue

        if not flat_phrases:
            raise RuntimeError(f"No valid phrases available to encode for split '{split}'")

        embeddings = encoder.encode(texts=flat_phrases, batch_size=batch_size, return_tokens=False)
        if embeddings.shape[0] != len(flat_phrases):
            raise RuntimeError(
                f"Embedding row count mismatch for split '{split}': "
                f"got {embeddings.shape[0]} vectors for {len(flat_phrases)} phrases"
            )

        for idx, sample_id in enumerate(ids):
            if sample_id in index_map:
                duplicate_ids.append(sample_id)
            index_map[sample_id] = {"split": split, "index": idx}

        payload: Dict[str, Any] = {
            "split": split,
            "prepared_root": str(root),
            "ids": ids,
            "embeddings": embeddings,
            "phrase_mask": torch.ones((len(flat_phrases),), dtype=torch.bool),
            "phrase_types": flat_phrase_types,
            "phrase_weights": torch.tensor(flat_phrase_weights, dtype=torch.float32),
            "token_target_indices": torch.tensor(flat_token_target_indices, dtype=torch.long),
            "sample_offsets": sample_offsets,
            "num_phrases_per_sample": [sample_offsets[i + 1] - sample_offsets[i] for i in range(len(ids))],
            "text_sources": text_sources,
            "model_meta": encoder.metadata(),
            "runtime_meta": runtime_meta,
            "preprocess_meta": {
                "empty_placeholder": empty_placeholder,
                "num_empty_fallback": num_empty_fallback,
                "num_phrase_fallback": num_phrase_fallback,
                "num_samples": len(ids),
                "num_phrases": len(flat_phrases),
                "phrase_level": False,
                "phrase_types": ["OBJECT"],
                "phrase_type_weights": {"OBJECT": 1.0},
                "spacy_model": "not_used_in_object_1to1",
                "max_phrases_per_sample": int(max_phrases_per_sample),
                "phrase_schema": "object_description_1to1_v1",
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if save_texts:
            payload["source_texts"] = source_texts
            payload["phrases"] = phrase_texts_by_sample

        torch.save(payload, output_file)
        split_stats[split] = {
            "status": "written",
            "output": str(output_file),
            "num_samples": len(ids),
            "num_phrases": len(flat_phrases),
            "num_empty_fallback": num_empty_fallback,
            "num_phrase_fallback": num_phrase_fallback,
        }

    index_file = ""
    if save_index:
        index_payload = {
            "prepared_root": str(root),
            "output_dir": str(out_dir),
            "index": index_map,
            "duplicate_ids": sorted(set(duplicate_ids)),
            "runtime_meta": runtime_meta,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        index_path = out_dir / "embeddings_index.pt"
        torch.save(index_payload, index_path)
        index_file = str(index_path)

    total_samples = sum(int(item.get("num_samples", 0)) for item in split_stats.values())
    total_phrases = sum(int(item.get("num_phrases", 0)) for item in split_stats.values())
    return {
        "prepared_dir": str(root),
        "output_dir": str(out_dir),
        "splits": split_stats,
        "total_samples": total_samples,
        "total_phrases": total_phrases,
        "duplicate_ids": sorted(set(duplicate_ids)),
        "index_file": index_file,
        "model_meta": encoder.metadata(),
        "runtime_meta": runtime_meta,
        "phrase_types": list(enabled_phrase_types),
        "phrase_type_weights": {k: float(v) for k, v in phrase_weights_by_type.items()},
        "spacy_model": str(spacy_model),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute OpenCLIP text embeddings from image-level sample metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prepared-dir", type=str, default="datasets/DIOR_RSVG", help="Prepared dataset root directory.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated split names.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for .pt embedding files.")
    parser.add_argument("--model-name", type=str, default="ViT-L-14", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", type=str, default="auto", help="Device for text encoding, e.g. 'auto', 'cpu', '0'.")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Text encoding precision.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for text encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split embedding files.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional max samples per split for debugging. 0 means all.")
    parser.add_argument("--empty-placeholder", type=str, default="[NO_DESCRIPTION]", help="Fallback text when description is empty.")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm", help="spaCy model name for phrase parsing.")
    parser.add_argument("--phrase-types", type=str, default="NP,PP,ADJP", help="Phrase types to extract, comma-separated.")
    parser.add_argument(
        "--phrase-type-weights",
        type=str,
        default="NP:1.0,PP:1.2,ADJP:0.8,FALLBACK:1.0",
        help="Comma-separated type:weight pairs for phrase aggregation priors.",
    )
    parser.add_argument("--max-phrases-per-sample", type=int, default=24, help="Maximum phrases retained per sample.")
    parser.add_argument("--no-save-texts", action="store_true", help="Do not store cleaned text strings in output .pt files.")
    parser.add_argument("--no-save-index", action="store_true", help="Do not generate merged embeddings_index.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = parse_splits(args.splits)

    result = build_openclip_text_embeddings(
        prepared_dir=args.prepared_dir,
        splits=splits,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        empty_placeholder=args.empty_placeholder,
        max_samples=args.max_samples,
        save_texts=not args.no_save_texts,
        save_index=not args.no_save_index,
        spacy_model=args.spacy_model,
        phrase_types=args.phrase_types,
        phrase_type_weights=_parse_phrase_weight_string(args.phrase_type_weights),
        max_phrases_per_sample=args.max_phrases_per_sample,
    )

    print("\n=== OpenCLIP Text Embedding Summary ===")
    print(f"prepared dir: {result['prepared_dir']}")
    print(f"output dir:   {result['output_dir']}")
    print(f"total samples: {result['total_samples']}")
    print(f"total phrases: {result['total_phrases']}")
    print(f"model meta: {result['model_meta']}")
    print(f"phrase types: {result['phrase_types']}")
    print(f"phrase type weights: {result['phrase_type_weights']}")
    print(f"spaCy model: {result['spacy_model']}")
    for split, stats in result["splits"].items():
        print(
            f"  {split}: status={stats['status']}, samples={stats['num_samples']}, phrases={stats.get('num_phrases', 0)}, "
            f"empty_fallback={stats['num_empty_fallback']}, phrase_fallback={stats.get('num_phrase_fallback', 0)}"
        )
        print(f"    output: {stats['output']}")

    if result.get("index_file"):
        print(f"index file: {result['index_file']}")
    if result["duplicate_ids"]:
        print(f"warning: duplicate ids across splits: {len(result['duplicate_ids'])}")


if __name__ == "__main__":
    main()
