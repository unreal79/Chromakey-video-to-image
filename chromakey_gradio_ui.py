# -*- coding: utf-8 -*-

""" Gradio UI for chromakey_video2png.py utility.
"""

from __future__ import annotations # For Python 3.10 compatibility with type hints in function signatures

import multiprocessing
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import gradio as gr

from chromakey_video2png import chromakey_image2png, chromakey_video2png, find_best


APP_ROOT = Path(__file__).resolve().parent
GRADIO_RUNS_ROOT = APP_ROOT / "gradio_runs"
FIND_BEST_ROOT = GRADIO_RUNS_ROOT / "find_best"
PROCESS_ROOT = GRADIO_RUNS_ROOT / "processed"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}
PRESET_ORDER = [("HSV", 0), ("HSV", 1), ("HSV", 2), ("BGR", 0), ("BGR", 1), ("BGR", 2)]
MAX_FRAME_INPUT = 5000
DEFAULT_COLOR_SPACE = "HSV"
DEFAULT_MASK_CHANNEL = 1
DEFAULT_PRESET = f"{DEFAULT_COLOR_SPACE} / channel {DEFAULT_MASK_CHANNEL}"
GALLERY_IDS = "#find-best-gallery, #process-preview-gallery"
GALLERY_SCOPE = f":is({GALLERY_IDS})"
FULLSCREEN_SCOPE = ":is(#find-best-gallery.gallery-browser-fullscreen, #process-preview-gallery.gallery-browser-fullscreen, #find-best-gallery:fullscreen, #process-preview-gallery:fullscreen)"
UI_HEAD = """
<script>
(() => {
  if (window.__chromakeyFullscreenHookInstalled) {
    return;
  }

  window.__chromakeyFullscreenHookInstalled = true;
  document.addEventListener(
    "click",
    (event) => {
      const fullscreenButton = event.target.closest('button[aria-label="Fullscreen"]');
            const exitButton = event.target.closest('button[aria-label="Exit fullscreen mode"], button[aria-label="Close"]');
            const galleryRoot = (fullscreenButton || exitButton)?.closest("#find-best-gallery, #process-preview-gallery");
            if (!galleryRoot) {
                return;
            }

            if (fullscreenButton) {
                galleryRoot.classList.add("gallery-browser-fullscreen");

                if (document.fullscreenElement === galleryRoot) {
                    return;
                }

                if (galleryRoot.requestFullscreen) {
                    galleryRoot.requestFullscreen().catch(() => {
                        galleryRoot.classList.add("gallery-browser-fullscreen");
                    });
                    return;
                }

                return;
            }

            galleryRoot.classList.remove("gallery-browser-fullscreen");
            if (document.fullscreenElement === galleryRoot && document.exitFullscreen) {
                document.exitFullscreen().catch(() => {});
            }
    },
    true,
  );

    document.addEventListener("fullscreenchange", () => {
        if (document.fullscreenElement) {
            return;
        }

        for (const node of document.querySelectorAll("#find-best-gallery, #process-preview-gallery")) {
            node.classList.remove("gallery-browser-fullscreen");
        }
    });
})();
</script>
"""
UI_CSS = f"""
{GALLERY_SCOPE} {{
    --checkerboard-pattern: conic-gradient(#b8b8b8 25%, #5f5f5f 0 50%, #b8b8b8 0 75%, #5f5f5f 0);
    background-image: var(--checkerboard-pattern);
    background-size: 24px 24px;
    border-radius: 14px;
    padding: 10px;
}}

{GALLERY_SCOPE} .gallery-container,
{GALLERY_SCOPE} .grid-wrap {{
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
}}

{GALLERY_SCOPE} .grid-container {{
    --object-fit: contain !important;
    gap: 10px !important;
}}

{GALLERY_SCOPE} .thumbnail-item.thumbnail-lg {{
    width: 100% !important;
    height: 150px !important;
    background: var(--checkerboard-pattern) 0 0 / 24px 24px !important;
}}

{GALLERY_SCOPE} .gallery-item img,
{GALLERY_SCOPE} .thumbnail-item.thumbnail-lg img {{
    width: 100% !important;
    height: calc(100% - 28px) !important;
    object-fit: contain !important;
    background: transparent !important;
}}

{GALLERY_SCOPE} .caption-label,
{GALLERY_SCOPE} .caption {{
    color: #161616 !important;
    background: rgba(242, 242, 242, 0.8) !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
}}

{GALLERY_SCOPE}:has(.grid-container) {{
    height: auto !important;
    overflow: visible !important;
}}

{GALLERY_SCOPE}:has(.preview) {{
    height: min(82vh, 980px) !important;
    overflow: hidden !important;
}}

{GALLERY_SCOPE} .gallery-container:has(.preview) {{
    min-height: min(78vh, 920px) !important;
}}

{GALLERY_SCOPE} .preview {{
    height: calc(100% - 104px) !important;
    min-height: 360px !important;
    display: flex !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    overflow: auto !important;
    background: var(--checkerboard-pattern) 0 0 / 24px 24px !important;
}}

{GALLERY_SCOPE} button[aria-label="detailed view of selected image"] {{
    width: max-content !important;
    height: auto !important;
    max-width: none !important;
    max-height: none !important;
    flex: 0 0 auto !important;
    align-self: flex-start !important;
    background: transparent !important;
}}

{GALLERY_SCOPE} .preview img {{
    width: auto !important;
    height: auto !important;
    max-width: none !important;
    max-height: none !important;
    object-fit: initial !important;
}}

{GALLERY_SCOPE} button[aria-label="detailed view of selected image"] img {{
    width: auto !important;
    height: auto !important;
    max-width: none !important;
    max-height: none !important;
    object-fit: initial !important;
}}

{GALLERY_SCOPE} .thumbnail-item.thumbnail-small {{
    width: 72px !important;
    height: 72px !important;
    background: var(--checkerboard-pattern) 0 0 / 24px 24px !important;
}}

{GALLERY_SCOPE} .thumbnail-item.thumbnail-small img {{
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}}

{FULLSCREEN_SCOPE} {{
    width: 100vw !important;
    height: 100vh !important;
    max-width: none !important;
    max-height: none !important;
    margin: 0 !important;
    padding: 20px !important;
    border-radius: 0 !important;
}}

{FULLSCREEN_SCOPE} .gallery-container:has(.preview) {{
    min-height: calc(100vh - 40px) !important;
}}

{FULLSCREEN_SCOPE} .preview {{
    height: calc(100vh - 160px) !important;
}}
"""


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def empty_state() -> dict:
    return {
        "source_path": None,
        "source_kind": None,
        "find_best_dir": None,
        "selected_preset": DEFAULT_PRESET,
        "mask_as_hsv": True,
        "mask_channel": DEFAULT_MASK_CHANNEL,
        "output_dir": None,
        "available_presets": [],
    }


def detect_source_kind(source_path: str) -> str:
    suffix = Path(source_path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise gr.Error(f"Unsupported file type: {suffix or 'missing extension'}")


def preset_caption(color_space: str, channel: int) -> str:
    return f"{color_space} / channel {channel}"


def parse_preset_caption(selection: Optional[str]) -> tuple[bool, int]:
    if not selection:
        raise gr.Error("Choose one of the six previews before continuing.")

    match = re.fullmatch(r"(HSV|BGR)\s*/\s*channel\s*(\d)", selection.strip())
    if not match:
        raise gr.Error(f"Could not parse preset selection: {selection}")

    return match.group(1) == "HSV", int(match.group(2))


def collect_find_best_outputs(output_dir: Path) -> list[tuple[str, str]]:
    collected: dict[tuple[str, int], tuple[str, str]] = {}
    for image_path in output_dir.glob("*.png"):
        if image_path.name.endswith("_mask.png"):
            continue

        match = re.search(r"(HSV|BGR)_(\d)\.png$", image_path.name)
        if not match:
            continue

        color_space = match.group(1)
        channel = int(match.group(2))
        collected[(color_space, channel)] = (str(image_path), preset_caption(color_space, channel))

    ordered = [collected[key] for key in PRESET_ORDER if key in collected]
    if len(ordered) != 6:
        raise gr.Error(f"Expected 6 preview files, found {len(ordered)} in {output_dir}")

    return ordered


def build_default_output_dir(source_path: str) -> str:
    stem = Path(source_path).stem or "chromakey"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return str(ensure_directory(PROCESS_ROOT) / f"{stem}_{run_id}")


def normalize_integer(value: float | int | None, minimum: int, maximum: int) -> int:
    if value is None:
        return minimum
    return max(minimum, min(maximum, int(round(float(value)))))


def normalize_blur(value: float | int | None) -> int:
    normalized = normalize_integer(value, 0, 31)
    if normalized > 0 and normalized % 2 == 0:
        normalized = normalized + 1 if normalized < 31 else normalized - 1
    return normalized


def run_find_best_step(source_path: str, state: dict):
    if not source_path:
        raise gr.Error("Upload an image or a video first.")

    source_kind = detect_source_kind(source_path)
    run_dir = ensure_directory(FIND_BEST_ROOT / uuid.uuid4().hex)
    find_best(source_path, str(run_dir), is_image=source_kind == "image")

    gallery_items = collect_find_best_outputs(run_dir)
    choices = [caption for _, caption in gallery_items]
    next_state = {
        **empty_state(),
        "source_path": source_path,
        "source_kind": source_kind,
        "find_best_dir": str(run_dir),
        "selected_preset": DEFAULT_PRESET,
        "mask_as_hsv": True,
        "mask_channel": DEFAULT_MASK_CHANNEL,
        "available_presets": choices,
        "output_dir": build_default_output_dir(source_path),
    }

    status = f"Generated 6 previews in {run_dir}"
    process_note = "Step 2 is ready. Apply a preset or tune values manually."

    return (
        next_state,
        status,
        gallery_items,
        gr.update(choices=choices, value=DEFAULT_PRESET, interactive=True),
        gr.update(visible=source_kind == "video"),
        next_state["output_dir"],
        process_note,
        [],
        DEFAULT_COLOR_SPACE,
        DEFAULT_MASK_CHANNEL,
    )


def apply_selected_preset(selection: Optional[str], state: dict):
    mask_as_hsv, mask_channel = parse_preset_caption(selection)
    next_state = {
        **state,
        "selected_preset": selection,
        "mask_as_hsv": mask_as_hsv,
        "mask_channel": mask_channel,
    }

    color_space = "HSV" if mask_as_hsv else "BGR"
    return next_state, color_space, mask_channel


def run_processing_step(
    state: dict,
    color_space: str,
    mask_channel: int,
    all_frames: bool,
    frames_value: float | int,
    margin_value: float | int,
    kernel_value: float | int,
    dilate_value: float | int,
    blur_value: float | int,
    mask_out: bool,
    post_process: bool,
    post_margin_value: float | int,
    output_dir_value: str,
    progress: gr.Progress = gr.Progress(),
):
    source_path = state.get("source_path")
    source_kind = state.get("source_kind")
    if not source_path or not source_kind:
        raise gr.Error("Run step 1 first so the interface knows what to process.")

    output_dir = Path(output_dir_value).expanduser() if output_dir_value else Path(build_default_output_dir(source_path))
    ensure_directory(output_dir)

    mask_as_hsv = color_space == "HSV"
    frames_max = -1 if all_frames or source_kind == "image" else normalize_integer(frames_value, 1, MAX_FRAME_INPUT)
    margin = normalize_integer(margin_value, 0, 255)
    kernel = normalize_integer(kernel_value, 0, 31)
    dilate = normalize_integer(dilate_value, 0, 31)
    blur = normalize_blur(blur_value)
    post_margin = normalize_integer(post_margin_value, 0, 255)
    mask_channel = normalize_integer(mask_channel, 0, 2)

    tracker = {
        "processed": 0,
        "total": 0,
        "preview_path": None,
        "error": None,
    }
    lock = threading.Lock()

    def on_progress(processed: int, total: int) -> None:
        with lock:
            tracker["processed"] = processed
            tracker["total"] = total

    def on_result(preview_path: str) -> None:
        with lock:
            if tracker["preview_path"] is None:
                tracker["preview_path"] = preview_path

    def worker() -> None:
        try:
            if source_kind == "video":
                chromakey_video2png(
                    path=source_path,
                    output_folder=str(output_dir),
                    mask_as_hsv=mask_as_hsv,
                    mask_channel=mask_channel,
                    margin=margin,
                    kernel_ones=kernel,
                    dilate=dilate,
                    blur=blur,
                    frames_max=frames_max,
                    mask_out=mask_out,
                    post_process=post_process,
                    post_process_margin=post_margin,
                    progress_callback=on_progress,
                    result_callback=on_result,
                )
            else:
                chromakey_image2png(
                    path=source_path,
                    output_folder=str(output_dir),
                    mask_as_hsv=mask_as_hsv,
                    mask_channel=mask_channel,
                    margin=margin,
                    kernel_ones=kernel,
                    dilate=dilate,
                    blur=blur,
                    mask_out=mask_out,
                    post_process=post_process,
                    post_process_margin=post_margin,
                    progress_callback=on_progress,
                    result_callback=on_result,
                )
        except Exception:
            with lock:
                tracker["error"] = traceback.format_exc()

    processing_thread = threading.Thread(target=worker, daemon=True)
    processing_thread.start()

    progress(0, desc="Preparing processing")
    preview_items: list[tuple[str, str]] = []
    yield f"Started processing into {output_dir}", preview_items, str(output_dir)

    while processing_thread.is_alive():
        with lock:
            snapshot = dict(tracker)

        if snapshot["preview_path"]:
            preview_items = [(snapshot["preview_path"], "First processed frame")]

        processed = int(snapshot["processed"] or 0)
        total = int(snapshot["total"] or 0)
        denominator = max(total, processed, 1)
        progress_value = min(processed / denominator, 0.99) if processed else 0.02
        progress_label = f"Processing {processed}/{total}" if total else f"Processing {processed} frame(s)"
        progress(progress_value, desc=progress_label)

        yield f"{progress_label}. Output folder: {output_dir}", preview_items, str(output_dir)
        time.sleep(0.2)

    processing_thread.join()
    with lock:
        snapshot = dict(tracker)

    if snapshot["preview_path"]:
        preview_items = [(snapshot["preview_path"], "First processed frame")]

    if snapshot["error"]:
        last_line = snapshot["error"].strip().splitlines()[-1]
        yield f"Processing failed: {last_line}", preview_items, str(output_dir)
        return

    processed = int(snapshot["processed"] or 0)
    total = int(snapshot["total"] or 0)
    progress(1.0, desc=f"Completed {processed}/{total}" if total else "Completed")
    yield f"Processing finished. Saved output to {output_dir}", preview_items, str(output_dir)


with gr.Blocks(
    title="Chromakey Video to Image",
) as demo:
    gr.Markdown(
        "# Chromakey Video to Image\n"
        "Interactive Gradio interface for the existing CLI utility."
    )

    app_state = gr.State(empty_state())

    with gr.Group():
        gr.Markdown("## 1. Find the best HSV/BGR preset")
        source_file = gr.File(
            label="Source image or video",
            file_types=sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS),
            type="filepath",
        )
        run_find_best_button = gr.Button("Generate 6 previews", variant="primary")
        trial_status = gr.Textbox(label="Step 1 status", interactive=False)
        find_best_gallery = gr.Gallery(
            label="Six preview images",
            columns=3,
            rows=2,
            height=360,
            object_fit="contain",
            elem_id="find-best-gallery",
            elem_classes=["checkerboard-frame"],
        )
        preferred_preview = gr.Radio(label="Preferred preview", choices=[], interactive=True)

    with gr.Group():
        gr.Markdown("## 2. Tune parameters and process")
        process_status = gr.Textbox(label="Processing status", interactive=False)
        output_dir = gr.Textbox(label="Output folder", placeholder="Leave empty to use an auto-generated folder")

        color_space = gr.Radio(label="Color space", choices=["HSV", "BGR"], value=DEFAULT_COLOR_SPACE)
        mask_channel = gr.Radio(label="Mask channel", choices=[0, 1, 2], value=DEFAULT_MASK_CHANNEL)

        with gr.Group(visible=False) as frames_group:
            all_frames = gr.Checkbox(label="Process all frames", value=False)
            frames_slider = gr.Slider(label="Frames to process", minimum=1, maximum=MAX_FRAME_INPUT, value=10, step=1)

        margin_slider = gr.Slider(label="Crop margin", minimum=0, maximum=255, value=50, step=1)

        kernel_slider = gr.Slider(label="Kernel size for morphology", minimum=0, maximum=31, value=3, step=1)

        dilate_slider = gr.Slider(label="Dilation iterations inside edges", minimum=0, maximum=31, value=1, step=1)

        blur_slider = gr.Slider(label="Median blur size for edges", minimum=0, maximum=31, value=5, step=1)

        with gr.Row():
            mask_out = gr.Checkbox(label="Save mask preview files", value=False)
            post_process = gr.Checkbox(label="Enable post-process edge desaturation", value=False)

        post_margin_slider = gr.Slider(label="Post-process edge desaturation margin", minimum=0, maximum=255,
                                       value=20, step=1)

        process_button = gr.Button("Process", variant="primary")
        process_preview = gr.Gallery(
            label="First output preview",
            columns=1,
            rows=1,
            height=320,
            object_fit="contain",
            elem_id="process-preview-gallery",
            elem_classes=["checkerboard-frame"],
        )

    run_find_best_button.click(
        fn=run_find_best_step,
        inputs=[source_file, app_state],
        outputs=[
            app_state,
            trial_status,
            find_best_gallery,
            preferred_preview,
            frames_group,
            output_dir,
            process_status,
            process_preview,
            color_space,
            mask_channel,
        ],
    )

    preferred_preview.change(
        fn=apply_selected_preset,
        inputs=[preferred_preview, app_state],
        outputs=[app_state, color_space, mask_channel],
    )

    process_button.click(
        fn=run_processing_step,
        inputs=[
            app_state,
            color_space,
            mask_channel,
            all_frames,
            frames_slider,
            margin_slider,
            kernel_slider,
            dilate_slider,
            blur_slider,
            mask_out,
            post_process,
            post_margin_slider,
            output_dir,
        ],
        outputs=[process_status, process_preview, output_dir],
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    ensure_directory(FIND_BEST_ROOT)
    ensure_directory(PROCESS_ROOT)
    demo.queue(default_concurrency_limit=1).launch(css=UI_CSS, head=UI_HEAD)
