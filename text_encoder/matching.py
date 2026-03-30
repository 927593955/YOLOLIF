from __future__ import annotations

from typing import List, Tuple

import torch


def _greedy_assignment(cost: torch.Tensor) -> Tuple[List[int], List[int]]:
    """Fallback one-to-one assignment when scipy is unavailable."""
    rows = int(cost.shape[0])
    cols = int(cost.shape[1])
    if rows <= 0 or cols <= 0:
        return [], []

    used_cols = set()
    row_idx: List[int] = []
    col_idx: List[int] = []

    # Greedy over rows keeps implementation dependency-free.
    for r in range(rows):
        best_c = -1
        best_v = None
        for c in range(cols):
            if c in used_cols:
                continue
            v = float(cost[r, c].item())
            if best_v is None or v < best_v:
                best_v = v
                best_c = c
        if best_c >= 0:
            used_cols.add(best_c)
            row_idx.append(r)
            col_idx.append(best_c)

    return row_idx, col_idx


def linear_sum_assignment_torch(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve linear assignment on a 2D cost matrix with scipy-first strategy."""
    if cost.ndim != 2:
        raise ValueError(f"Expected 2D cost matrix, got shape={tuple(cost.shape)}")

    rows = int(cost.shape[0])
    cols = int(cost.shape[1])
    if rows <= 0 or cols <= 0:
        empty = torch.zeros((0,), device=cost.device, dtype=torch.long)
        return empty, empty

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        r_np, c_np = linear_sum_assignment(cost.detach().cpu().numpy())
        r = torch.as_tensor(r_np, device=cost.device, dtype=torch.long)
        c = torch.as_tensor(c_np, device=cost.device, dtype=torch.long)
        return r, c
    except Exception:
        r_list, c_list = _greedy_assignment(cost)
        r = torch.tensor(r_list, device=cost.device, dtype=torch.long)
        c = torch.tensor(c_list, device=cost.device, dtype=torch.long)
        return r, c


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU matrix for xyxy boxes: [N,4] x [M,4] -> [N,M]."""
    if boxes1.ndim != 2 or boxes2.ndim != 2 or int(boxes1.shape[1]) != 4 or int(boxes2.shape[1]) != 4:
        raise ValueError("Expected boxes as [N,4] and [M,4] in xyxy format.")

    n = int(boxes1.shape[0])
    m = int(boxes2.shape[0])
    if n <= 0 or m <= 0:
        return torch.zeros((n, m), device=boxes1.device, dtype=boxes1.dtype)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)).clamp(min=1e-6)
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)).clamp(min=1e-6)
    union = (area1[:, None] + area2[None, :] - inter).clamp(min=1e-6)
    return inter / union


def assign_by_score_matrix(scores: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-to-one assignment maximizing score matrix [K,M]."""
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape={tuple(scores.shape)}")
    if int(scores.shape[0]) == 0 or int(scores.shape[1]) == 0:
        empty = torch.zeros((0,), device=scores.device, dtype=torch.long)
        return empty, empty

    t = float(max(1e-3, temperature))
    cost = -(scores / t)
    return linear_sum_assignment_torch(cost)


def score_matrix_from_phrase_maps(
    phrase_logits_levels: List[torch.Tensor],
    image_index: int,
    token_indices: torch.Tensor,
    candidate_boxes_xyxy: torch.Tensor,
    input_hw: Tuple[int, int],
) -> torch.Tensor:
    """Build score matrix [K, M] by ROI-mean pooling phrase maps inside candidate boxes.

    Args:
        phrase_logits_levels: list of [B, T, H, W]
        image_index: index in current batch
        token_indices: [K] token ids to score
        candidate_boxes_xyxy: [M, 4] boxes in input-image scale
        input_hw: (H, W) of network input
    """
    if int(token_indices.numel()) <= 0 or int(candidate_boxes_xyxy.shape[0]) <= 0:
        return torch.zeros(
            (int(token_indices.numel()), int(candidate_boxes_xyxy.shape[0])),
            device=candidate_boxes_xyxy.device,
            dtype=candidate_boxes_xyxy.dtype,
        )

    in_h, in_w = int(input_hw[0]), int(input_hw[1])
    k = int(token_indices.numel())
    m = int(candidate_boxes_xyxy.shape[0])
    score = torch.zeros((k, m), device=candidate_boxes_xyxy.device, dtype=candidate_boxes_xyxy.dtype)
    level_count = 0

    for lvl in phrase_logits_levels:
        if not isinstance(lvl, torch.Tensor) or lvl.ndim != 4:
            continue
        if image_index < 0 or image_index >= int(lvl.shape[0]):
            continue

        _, t, h, w = lvl.shape
        idx = token_indices.clamp(min=0, max=max(0, t - 1)).to(device=lvl.device, dtype=torch.long)
        maps = lvl[image_index, idx, :, :]  # [K, H, W]

        # Map input-scale boxes to feature-map scale.
        x1 = (candidate_boxes_xyxy[:, 0] / max(float(in_w), 1.0) * float(w)).floor().to(dtype=torch.long)
        y1 = (candidate_boxes_xyxy[:, 1] / max(float(in_h), 1.0) * float(h)).floor().to(dtype=torch.long)
        x2 = (candidate_boxes_xyxy[:, 2] / max(float(in_w), 1.0) * float(w)).ceil().to(dtype=torch.long)
        y2 = (candidate_boxes_xyxy[:, 3] / max(float(in_h), 1.0) * float(h)).ceil().to(dtype=torch.long)

        x1 = x1.clamp(0, max(0, w - 1))
        y1 = y1.clamp(0, max(0, h - 1))
        x2 = x2.clamp(1, max(1, w))
        y2 = y2.clamp(1, max(1, h))

        lvl_score = torch.zeros((k, m), device=maps.device, dtype=maps.dtype)
        for j in range(m):
            xa = int(min(x1[j].item(), x2[j].item() - 1))
            ya = int(min(y1[j].item(), y2[j].item() - 1))
            xb = int(max(xa + 1, x2[j].item()))
            yb = int(max(ya + 1, y2[j].item()))
            roi = maps[:, ya:yb, xa:xb]
            if int(roi.numel()) <= 0:
                continue
            lvl_score[:, j] = roi.mean(dim=(1, 2))

        score = score + lvl_score.to(device=score.device, dtype=score.dtype)
        level_count += 1

    if level_count > 0:
        score = score / float(level_count)
    return score
