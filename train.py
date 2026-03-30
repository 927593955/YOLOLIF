import argparse
from pathlib import Path
from typing import Any, Dict
from ultralytics import YOLO

from dataset.utils import prepare_dataset
from lib.general import print_metrics, resolve_device, resolve_local_weights, _parse_phrase_types, _parse_phrase_weight_string
from text_encoder import TextGuidedDetectionTrainer, TextGuidedDetectionValidator, configure_text_guidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv12 model with common settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights","--model",dest="weights",type=str,default="./pretrain_model/yolov12m.pt",help="Local pretrained weights path, e.g. ./weights/yolov12n.pt",)
    parser.add_argument("--data", type=str, default="", help="Optional existing data.yaml path. If set, skip dataset preparation.")
    parser.add_argument("--voc-root", type=str, default="/data/bxc/OPT-RSVG", help="VOC root with JPGEImages/JPEGImages, Annotations and split txt files.")
    parser.add_argument("--images-dir", type=str, default="JPGEImages", help="Image folder name under VOC root.")
    parser.add_argument("--annotations-dir", type=str, default="Annotations", help="XML annotation folder name.")
    parser.add_argument("--train-list", type=str, default="train.txt", help="Train split txt file name.")
    parser.add_argument("--val-list", type=str, default="val.txt", help="Val split txt file name.")
    parser.add_argument("--test-list", type=str, default="test.txt", help="Test split txt file name.")
    parser.add_argument("--prepared-dir", "--prepared_dir", dest="prepared_dir", type=str, default="pre_datasets/OPT_RSVG", help="Output dir for generated data.yaml and image-xml mapping files.")
    
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=24, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Train image size.")
    parser.add_argument("--device", type=str, default="0", help="Device, e.g. '0', '0,1', 'cpu' or 'auto'.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker count.")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory.")
    parser.add_argument("--name", type=str, default="OPT-RSVG", help="Experiment name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse existing experiment directory.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    parser.add_argument("--run-val", action="store_true", help="Run an extra validation after training.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON during extra validation.")

    parser.add_argument("--text-embedding-dir", type=str, default="", help="Directory that contains *_text_embeddings.pt files. Empty means --prepared-dir.")
    parser.add_argument("--text-embedding-dim", type=int, default=768, help="Input text embedding dimension.")
    parser.add_argument("--text-phrase-types", type=str, default="NP,PP,ADJP", help="Phrase types used in preprocessing, comma-separated.")
    parser.add_argument("--text-phrase-type-weights",type=str,default="NP:1.0,PP:1.2,ADJP:0.8,FALLBACK:1.0",help="Phrase type priors for weighted phrase aggregation.",)
    parser.add_argument("--text-aggregation-mode",type=str,choices=("lse", "weighted_sum", "mean"),default="lse",help="Phrase-to-heatmap aggregation strategy.",)
    parser.add_argument("--text-guidance-strength", type=float, default=0.25, help="Feature modulation strength for text guidance.")
    parser.add_argument("--text-cls-gate-strength", type=float, default=0.8, help="Logit gating strength applied to the classification branch.")
    parser.add_argument("--text-cls-fusion-mode",type=str,choices=("additive", "multiplicative"),default="additive",help="How text guidance is fused into classification logits.",)
    parser.add_argument("--text-cls-gate-temperature",type=float,default=1.0,help="Temperature for transforming text gate logits before cls fusion.",)
    parser.add_argument("--text-cls-gate-bias-cap",type=float,default=1.0,help="Clamp magnitude for cls gate bias term (0 disables clamping).",)
    parser.add_argument("--text-allow-negative-cls-bias",action="store_false",dest="text_cls_gate_nonnegative",help="Allow text cls gate bias to be negative.",)
    parser.add_argument("--text-alignment-temperature", type=float, default=0.07, help="Temperature used in phrase-visual similarity computation.")
    parser.add_argument("--text-fuse-temperature", type=float, default=0.5, help="Temperature used in log-sum-exp phrase fusion.")
    parser.add_argument("--text-lambda-heatmap", type=float, default=0.3, help="Phrase contrastive supervision weight.")
    parser.add_argument("--text-lambda-phrase", type=float, default=0.2, help="Phrase-level supervision weight.")
    parser.add_argument("--text-lambda-set", type=float, default=0.6, help="Set matching supervision weight.")
    parser.add_argument("--text-matching-temperature", type=float, default=0.7, help="Temperature for set matching logits.")
    parser.add_argument("--text-seq-enhance", action="store_true", help="Enable lightweight text sequence enhancement before similarity.")
    parser.add_argument("--text-seq-conv-layers", type=int, default=1, help="Number of 1D conv residual layers for text sequence enhancement.")
    parser.add_argument("--text-seq-kernel-size", type=int, default=3, help="Kernel size for text sequence 1D conv.")
    parser.add_argument("--text-seq-dropout", type=float, default=0.0, help="Dropout ratio inside text sequence enhancement block.")
    parser.add_argument("--text-seq-pooling-mode",type=str,choices=("none", "learnable_weight"),default="none",help="Adaptive token weighting mode used before phrase aggregation.",)
    parser.add_argument("--text-seq-pool-temperature", type=float, default=1.0, help="Temperature for adaptive token weighting.")
    parser.add_argument("--visual-attr-enabled", type=bool, default=True, help="Enable normalized geometry/stat attributes on visual features before similarity.")
    parser.add_argument("--visual-attr-scale", type=float, default=1.0, help="Scale factor for visual attribute channels.")
    parser.add_argument("--visual-attr-eps", type=float, default=1e-6, help="Numerical epsilon used in visual attribute normalization.")
    parser.add_argument("--text-multi-proj-enabled", type=bool, default=True, help="Enable three parallel text-visual projections (geo/attr/sem) before similarity.")
    parser.add_argument("--text-multi-proj-score-scale", type=float, default=1.0, help="Scale factor applied after summing three projection scores.")
    parser.add_argument("--text-orth-loss-weight", type=float, default=0.05, help="Weight for orthogonality regularizer across three visual projections.")
    parser.add_argument("--strict-match-iou", type=float, default=0.5, help="IoU threshold used by strict 1:1 grounding metrics.")
    parser.add_argument("--strict-nms-conf", type=float, default=0.25, help="Confidence threshold for strict grounding candidate boxes.")
    parser.add_argument("--strict-max-candidates-factor",type=int,default=4,help="Limit strict grounding candidates to max(descriptions, factor * gt_count).",)
    parser.add_argument("--no-strict-grounding", action="store_false", dest="strict_grounding", help="Disable strict 1:1 grounding metrics.")
    parser.set_defaults(strict_grounding=True)
    parser.add_argument("--text-keep-augmentations", action="store_false", dest="text_disable_augmentations", help="Keep original augmentations in text-guided training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    data_path = str(Path(args.data).expanduser().resolve()) if str(args.data).strip() else prepare_dataset(args, device)
    weights_path = resolve_local_weights(args.weights)
    print(f"Using local pretrained weights: {weights_path}")

    model = YOLO(weights_path)
    print(f"Using device: {device}")

    embedding_root = Path(args.text_embedding_dir or args.prepared_dir).expanduser().resolve()
    phrase_types = _parse_phrase_types(args.text_phrase_types)
    phrase_type_weights = _parse_phrase_weight_string(args.text_phrase_type_weights)
    configure_text_guidance(
        {
            "enabled": True,
            "embedding_dir": str(embedding_root),
            "embedding_dim": int(args.text_embedding_dim),
            "phrase_types": phrase_types,
            "phrase_type_weights": phrase_type_weights,
            "aggregation_mode": str(args.text_aggregation_mode),
            "guidance_strength": float(args.text_guidance_strength),
            "cls_gate_strength": float(args.text_cls_gate_strength),
            "cls_fusion_mode": str(args.text_cls_fusion_mode),
            "cls_gate_nonnegative": bool(args.text_cls_gate_nonnegative),
            "cls_gate_temperature": float(args.text_cls_gate_temperature),
            "cls_gate_bias_cap": float(args.text_cls_gate_bias_cap),
            "alignment_temperature": float(args.text_alignment_temperature),
            "fuse_temperature": float(args.text_fuse_temperature),
            "lambda_heatmap": float(args.text_lambda_heatmap),
            "lambda_phrase": float(args.text_lambda_phrase),
            "lambda_set": float(args.text_lambda_set),
            "matching_temperature": float(args.text_matching_temperature),
            "strict_grounding": bool(args.strict_grounding),
            "strict_match_iou": float(args.strict_match_iou),
            "strict_nms_conf": float(args.strict_nms_conf),
            "strict_max_candidates_factor": int(args.strict_max_candidates_factor),
            "contrastive_loss_type": "logsigmoid_margin",
            "disable_augmentations": bool(args.text_disable_augmentations),
            "text_seq_enhance": bool(args.text_seq_enhance),
            "text_seq_conv_layers": int(args.text_seq_conv_layers),
            "text_seq_kernel_size": int(args.text_seq_kernel_size),
            "text_seq_dropout": float(args.text_seq_dropout),
            "text_seq_pooling_mode": str(args.text_seq_pooling_mode),
            "text_seq_pool_temperature": float(args.text_seq_pool_temperature),
            "visual_attr_enabled": bool(args.visual_attr_enabled),
            "visual_attr_scale": float(args.visual_attr_scale),
            "visual_attr_eps": float(args.visual_attr_eps),
            "multi_proj_enabled": bool(args.text_multi_proj_enabled),
            "multi_proj_score_scale": float(args.text_multi_proj_score_scale),
            "orth_loss_weight": float(args.text_orth_loss_weight),
        }
    )
    print(f"Text guidance enabled, embedding root: {embedding_root}")
    print(
        "Text phrase config: "
        f"types={phrase_types}, aggregation={args.text_aggregation_mode}, "
        f"type_weights={phrase_type_weights if phrase_type_weights else '<default>'}"
    )
    print(
        "Text cls fusion config: "
        f"mode={args.text_cls_fusion_mode}, strength={args.text_cls_gate_strength}, "
        f"nonnegative={args.text_cls_gate_nonnegative}, "
        f"temperature={args.text_cls_gate_temperature}, cap={args.text_cls_gate_bias_cap}"
    )
    print(
        "Text/visual enhancement config: "
        f"text_seq={args.text_seq_enhance}({args.text_seq_conv_layers}x{args.text_seq_kernel_size}, "
        f"pool={args.text_seq_pooling_mode}), "
        f"visual_attr={args.visual_attr_enabled}(scale={args.visual_attr_scale}), "
        f"multi_proj={args.text_multi_proj_enabled}(score_scale={args.text_multi_proj_score_scale}, "
        f"orth_w={args.text_orth_loss_weight})"
    )

    train_kwargs: Dict[str, Any] = {
        "data": data_path,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "seed": args.seed,
        "resume": args.resume,
        "cache": args.cache,
    }
    train_kwargs["device"] = device


    train_metrics = model.train(trainer=TextGuidedDetectionTrainer, **train_kwargs)
    print_metrics("Train Metrics", train_metrics)

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best = getattr(trainer, "best", None)
        last = getattr(trainer, "last", None)
        if isinstance(best, Path):
            print(f"best checkpoint: {best}")
        if isinstance(last, Path):
            print(f"last checkpoint: {last}")

    if args.run_val:
        print("Running extra validation...")
        val_kwargs: Dict[str, Any] = {
            "data": data_path,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": f"{args.name}-val",
            "exist_ok": args.exist_ok,
            "save_json": args.save_json,
        }
        val_kwargs["device"] = device

        val_metrics = model.val(validator=TextGuidedDetectionValidator, **val_kwargs)
        print_metrics("Validation Metrics", val_metrics)


if __name__ == "__main__":
    main()
