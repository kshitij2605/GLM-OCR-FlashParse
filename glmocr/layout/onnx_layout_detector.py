"""PP-DocLayoutV3 layout detector with ONNX Runtime backend.

Exports the PyTorch model to ONNX on first start, then uses a single
ORT InferenceSession for all subsequent inference. Thread-safe without
an external lock (ORT handles internal serialization).

Benefits over PyTorch backend:
- ~30% faster per-inference (0.060s vs 0.086s per batch)
- Thread-safe by design — no _layout_lock needed
- ~150-200 MB VRAM vs 130 MB for PyTorch (single instance)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessorFast,
)

from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
from glmocr.utils.logging import get_logger
from glmocr.utils.visualization_utils import save_layout_visualization

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)

# ONNX outputs we need, in order. Mapped from unnamed TorchScript trace outputs.
_ONNX_OUTPUT_NAMES = ["logits", "pred_boxes", "order_logits", "out_masks"]


class PPDocLayoutDetectorONNX(BaseLayoutDetector):
    """PP-DocLayoutV3 layout detector using ONNX Runtime.

    On start(), exports the PyTorch model to ONNX (cached on disk),
    creates an ORT InferenceSession, and frees the PyTorch model.
    """

    def __init__(self, config: "LayoutConfig"):
        super().__init__(config)

        self.model_dir = config.model_dir
        self.cuda_visible_devices = config.cuda_visible_devices

        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.batch_size = config.batch_size

        self.label_task_mapping = config.label_task_mapping
        self.id2label = config.id2label

        self._image_processor = None
        self._ort_session = None
        self._device = None
        self._onnx_path = None
        self._input_names = None

    def _get_onnx_cache_path(self) -> Path:
        """Get path for cached ONNX model, keyed by model directory."""
        model_hash = hashlib.md5(self.model_dir.encode()).hexdigest()[:8]
        cache_dir = Path(os.environ.get("GLMOCR_ONNX_CACHE", "/tmp/glmocr_onnx"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"pp_doclayout_v3_{model_hash}.onnx"

    def _export_onnx(self, onnx_path: Path) -> None:
        """Export PyTorch model to ONNX with dynamic batch size."""
        logger.info("Exporting PP-DocLayoutV3 to ONNX (one-time)...")

        model = PPDocLayoutV3ForObjectDetection.from_pretrained(self.model_dir)
        model.eval()

        if torch.cuda.is_available():
            device = (
                f"cuda:{self.cuda_visible_devices}"
                if self.cuda_visible_devices is not None
                else "cuda"
            )
        else:
            device = "cpu"
        model = model.to(device)

        if self.id2label is None:
            self.id2label = model.config.id2label

        # Create dummy input (batch=2 for tracing, dynamic axes handle variable batch)
        dummy_images = [
            Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8))
            for _ in range(2)
        ]
        dummy_inputs = self._image_processor(images=dummy_images, return_tensors="pt")
        dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}

        input_names = list(dummy_inputs.keys())

        # Run once to discover output order
        with torch.no_grad():
            test_out = model(**dummy_inputs)

        # Map output fields to their shapes for identification
        output_fields = ["logits", "pred_boxes", "order_logits", "out_masks"]
        output_names = []
        for field in output_fields:
            val = getattr(test_out, field, None)
            if val is not None:
                output_names.append(field)

        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

        onnx_size = onnx_path.stat().st_size / 1e6
        logger.info("ONNX export complete: %.1f MB → %s", onnx_size, onnx_path)

        # Free PyTorch model
        del model, dummy_inputs, test_out
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    def start(self):
        """Load image processor, export ONNX if needed, create ORT session."""
        import onnxruntime as ort

        logger.debug("Initializing PP-DocLayoutV3 (ONNX backend)...")

        self._image_processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(
            self.model_dir
        )

        # Determine device
        if torch.cuda.is_available():
            self._device = (
                f"cuda:{self.cuda_visible_devices}"
                if self.cuda_visible_devices is not None
                else "cuda"
            )
        else:
            self._device = "cpu"

        # Export ONNX if not cached
        onnx_path = self._get_onnx_cache_path()
        if not onnx_path.exists():
            self._export_onnx(onnx_path)
        else:
            logger.info("Using cached ONNX model: %s", onnx_path)
            # Still need id2label from config or model
            if self.id2label is None:
                model = PPDocLayoutV3ForObjectDetection.from_pretrained(self.model_dir)
                self.id2label = model.config.id2label
                del model
                if self._device.startswith("cuda"):
                    torch.cuda.empty_cache()

        self._onnx_path = onnx_path

        # Create ORT session
        providers = []
        if self._device.startswith("cuda"):
            device_id = 0
            if self.cuda_visible_devices is not None:
                device_id = int(self.cuda_visible_devices)
            providers.append(
                ("CUDAExecutionProvider", {"device_id": device_id})
            )
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._ort_session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers
        )

        # Store input names
        self._input_names = [inp.name for inp in self._ort_session.get_inputs()]

        active_provider = self._ort_session.get_providers()[0]
        logger.info(
            "PP-DocLayoutV3 ONNX loaded on %s (provider: %s)",
            self._device,
            active_provider,
        )

    def stop(self):
        """Unload ORT session."""
        self._ort_session = None
        self._image_processor = None
        self._device = None
        self._input_names = None
        logger.debug("PP-DocLayoutV3 ONNX stopped.")

    def _run_ort(self, images: list[Image.Image]) -> SimpleNamespace:
        """Run ORT inference, return outputs as a SimpleNamespace with torch tensors."""
        inputs = self._image_processor(images=images, return_tensors="pt")
        input_feed = {k: v.numpy() for k, v in inputs.items()}

        ort_outputs = self._ort_session.run(None, input_feed)

        # Map ORT outputs to named fields as torch tensors on device
        output_names = [o.name for o in self._ort_session.get_outputs()]
        result = {}
        for name, arr in zip(output_names, ort_outputs):
            result[name] = torch.from_numpy(arr).to(self._device)

        return SimpleNamespace(**result)

    def _apply_per_class_threshold(self, raw_results: List[Dict]):
        """Filter detections by per-class confidence thresholds."""
        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}
        class_thresholds = {}
        for key, value in self.threshold_by_class.items():
            if isinstance(key, str):
                if key in label2id:
                    class_thresholds[label2id[key]] = float(value)
            else:
                class_thresholds[int(key)] = float(value)

        fallback = self.threshold
        filtered = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]
            thresholds = torch.full_like(scores, fallback)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh
            keep = scores >= thresholds
            new_result = {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": result["boxes"][keep],
            }
            if "order_seq" in result:
                new_result["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                new_result["polygon_points"] = [
                    p for p, k in zip(result["polygon_points"], keep_list) if k
                ]
            filtered.append(new_result)
        return filtered

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        visualization_output_dir: Optional[str] = None,
        global_start_idx: int = 0,
    ) -> List[List[Dict]]:
        """Batch-detect layout regions using ONNX Runtime.

        Same interface as PPDocLayoutDetector.process().
        """
        if self._ort_session is None:
            raise RuntimeError("Layout detector not started. Call start() first.")

        num_images = len(images)
        image_batch = []
        for image in images:
            image_width, image_height = image.size
            image_array = np.array(image.convert("RGB"))
            image_batch.append((image_array, image_width, image_height))

        pil_images = [Image.fromarray(img[0]) for img in image_batch]
        all_paddle_format_results = []

        for chunk_start in range(0, num_images, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_images)
            chunk_pil = pil_images[chunk_start:chunk_end]

            # Run ONNX inference
            outputs = self._run_ort(chunk_pil)

            target_sizes = torch.tensor(
                [img.size[::-1] for img in chunk_pil], device=self._device
            )

            # Pre-filter tiny boxes (same logic as PyTorch detector)
            try:
                if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
                    pred_boxes = outputs.pred_boxes
                    if hasattr(outputs, "out_masks") and outputs.out_masks is not None:
                        mask_h, mask_w = outputs.out_masks.shape[-2:]
                    else:
                        mask_h, mask_w = 200, 200
                    min_norm_w = 1.0 / mask_w
                    min_norm_h = 1.0 / mask_h
                    box_wh = pred_boxes[..., 2:4]
                    valid_mask = (box_wh[..., 0] > min_norm_w) & (
                        box_wh[..., 1] > min_norm_h
                    )
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        invalid_mask = ~valid_mask
                        if invalid_mask.any():
                            outputs.logits.masked_fill_(
                                invalid_mask.unsqueeze(-1), -100.0
                            )
            except Exception as e:
                logger.warning("Pre-filter failed (%s), continuing...", e)

            if self.threshold_by_class:
                pre_threshold = min(
                    self.threshold, min(self.threshold_by_class.values())
                )
            else:
                pre_threshold = self.threshold

            # Post-process using HuggingFace processor
            try:
                raw_results = self._image_processor.post_process_object_detection(
                    outputs,
                    threshold=pre_threshold,
                    target_sizes=target_sizes,
                )
            except Exception as e:
                logger.warning(
                    "Layout post_process failed for chunk (retrying per-image): %s", e
                )
                raw_results = []
                for i, img in enumerate(chunk_pil):
                    try:
                        single_out = self._run_ort([img])
                        single_target = torch.tensor(
                            [img.size[::-1]], device=self._device
                        )
                        single_result = (
                            self._image_processor.post_process_object_detection(
                                single_out,
                                threshold=pre_threshold,
                                target_sizes=single_target,
                            )
                        )
                        raw_results.append(single_result[0])
                    except Exception as e2:
                        logger.warning(
                            "Layout post_process failed for image %s: %s",
                            chunk_start + i, e2,
                        )
                        raw_results.append({
                            "scores": torch.tensor([], device=self._device),
                            "labels": torch.tensor(
                                [], dtype=torch.long, device=self._device
                            ),
                            "boxes": torch.tensor(
                                [], device=self._device
                            ).reshape(0, 4),
                            "order_seq": torch.tensor(
                                [], dtype=torch.long, device=self._device
                            ),
                        })

            if self.threshold_by_class:
                raw_results = self._apply_per_class_threshold(raw_results)

            img_sizes = [img.size for img in chunk_pil]
            paddle_format_results = apply_layout_postprocess(
                raw_results=raw_results,
                id2label=self.id2label,
                img_sizes=img_sizes,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
            all_paddle_format_results.extend(paddle_format_results)

        # Visualization
        saved_vis_paths = []
        if save_visualization and visualization_output_dir:
            vis_output_path = Path(visualization_output_dir)
            vis_output_path.mkdir(parents=True, exist_ok=True)
            for img_idx, img_results in enumerate(all_paddle_format_results):
                vis_img = np.array(pil_images[img_idx])
                save_filename = f"layout_page{global_start_idx + img_idx}.jpg"
                save_path = vis_output_path / save_filename
                save_layout_visualization(
                    image=vis_img,
                    boxes=img_results,
                    save_path=str(save_path),
                    show_label=True,
                    show_score=True,
                    show_index=True,
                )
                saved_vis_paths.append(str(save_path))

        # Convert to final output format
        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width = image_batch[img_idx][1]
            image_height = image_batch[img_idx][2]
            results = []
            valid_index = 0
            for item in paddle_results:
                label = item["label"]
                score = item["score"]
                box = item["coordinate"]
                task_type = None
                for task_item, labels in self.label_task_mapping.items():
                    if isinstance(labels, list) and label in labels:
                        task_type = task_item
                        break
                if task_type is None or task_type == "abandon":
                    continue
                x1, y1, x2, y2 = box
                x1_norm = int(float(x1) / image_width * 1000)
                y1_norm = int(float(y1) / image_height * 1000)
                x2_norm = int(float(x2) / image_width * 1000)
                y2_norm = int(float(y2) / image_height * 1000)

                poly_array = item["polygon_points"]
                polygon = [
                    [
                        int(float(point[0]) / image_width * 1000),
                        int(float(point[1]) / image_height * 1000),
                    ]
                    for point in poly_array
                ]

                results.append(
                    {
                        "index": valid_index,
                        "label": label,
                        "score": float(score),
                        "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],
                        "polygon": polygon,
                        "task_type": task_type,
                    }
                )
                valid_index += 1
            all_results.append(results)

        return all_results
