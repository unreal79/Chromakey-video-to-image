# -*- coding: utf-8 -*-

""" Gradio UI for chromakey_video2png utility.
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


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def empty_state() -> dict:
    return {
        "source_path": None,
        "source_kind": None,
        "find_best_dir": None,
        "selected_preset": None,
        "mask_as_hsv": True,
        "mask_channel": 1,
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


def slider_to_number(value: float | int | None, minimum: int, maximum: int) -> int:
    return normalize_integer(value, minimum, maximum)


def number_to_slider(value: float | int | None, minimum: int, maximum: int) -> tuple[int, int]:
    normalized = normalize_integer(value, minimum, maximum)
    return normalized, normalized


def blur_slider_to_number(value: float | int | None) -> int:
    return normalize_blur(value)


def blur_number_to_slider(value: float | int | None) -> tuple[int, int]:
    normalized = normalize_blur(value)
    return normalized, normalized


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
        "available_presets": choices,
        "output_dir": build_default_output_dir(source_path),
    }

    source_info = f"Loaded {source_kind}: {Path(source_path).name}"
    status = f"Generated 6 previews in {run_dir}"
    selection_help = "Review the gallery and choose the best preview below."
    process_note = "Step 2 is ready. Apply a preset or tune values manually."

    return (
        next_state,
        status,
        gallery_items,
        gr.update(choices=choices, value=None, interactive=True),
        selection_help,
        source_info,
        gr.update(visible=source_kind == "video"),
        next_state["output_dir"],
        process_note,
        [],
    )


def apply_selected_preset(selection: Optional[str], state: dict):
    mask_as_hsv, mask_channel = parse_preset_caption(selection)
    next_state = {
        **state,
        "selected_preset": selection,
        "mask_as_hsv": mask_as_hsv,
        "mask_channel": mask_channel,
    }

    summary = f"Selected preset: {selection}"
    color_space = "HSV" if mask_as_hsv else "BGR"
    return next_state, summary, color_space, mask_channel


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


with gr.Blocks(title="Chromakey Video to Image") as demo:
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
        source_info = gr.Textbox(label="Loaded source", interactive=False)
        find_best_gallery = gr.Gallery(
            label="Six preview images",
            columns=3,
            rows=2,
            height=420,
        )
        preferred_preview = gr.Radio(label="Preferred preview", choices=[], interactive=True)
        selection_status = gr.Textbox(label="Chosen preset", interactive=False)

    with gr.Group():
        gr.Markdown("## 2. Tune parameters and process")
        process_status = gr.Textbox(label="Processing status", interactive=False)
        output_dir = gr.Textbox(label="Output folder", placeholder="Leave empty to use an auto-generated folder")

        with gr.Row():
            color_space = gr.Radio(label="Color space", choices=["HSV", "BGR"], value="HSV")
            mask_channel = gr.Dropdown(label="Mask channel", choices=[0, 1, 2], value=1)

        with gr.Group(visible=False) as frames_group:
            all_frames = gr.Checkbox(label="Process all frames", value=False)
            with gr.Row():
                frames_slider = gr.Slider(label="Frames to process", minimum=1, maximum=MAX_FRAME_INPUT, value=10, step=1)
                frames_number = gr.Number(label="Frames input", value=10, precision=0)

        with gr.Row():
            margin_slider = gr.Slider(label="Margin", minimum=0, maximum=255, value=50, step=1)
            margin_number = gr.Number(label="Margin input", value=50, precision=0)

        with gr.Row():
            kernel_slider = gr.Slider(label="Kernel", minimum=0, maximum=31, value=3, step=1)
            kernel_number = gr.Number(label="Kernel input", value=3, precision=0)

        with gr.Row():
            dilate_slider = gr.Slider(label="Dilate", minimum=0, maximum=31, value=1, step=1)
            dilate_number = gr.Number(label="Dilate input", value=1, precision=0)

        with gr.Row():
            blur_slider = gr.Slider(label="Blur", minimum=0, maximum=31, value=5, step=1)
            blur_number = gr.Number(label="Blur input", value=5, precision=0)

        with gr.Row():
            mask_out = gr.Checkbox(label="Save mask preview files", value=False)
            post_process = gr.Checkbox(label="Enable post-process desaturation", value=False)

        with gr.Row():
            post_margin_slider = gr.Slider(label="Post-process margin", minimum=0, maximum=255, value=20, step=1)
            post_margin_number = gr.Number(label="Post-process margin input", value=20, precision=0)

        process_button = gr.Button("Process", variant="primary")
        process_preview = gr.Gallery(
            label="First output preview",
            columns=1,
            rows=1,
            height=320,
        )

    run_find_best_button.click(
        fn=run_find_best_step,
        inputs=[source_file, app_state],
        outputs=[
            app_state,
            trial_status,
            find_best_gallery,
            preferred_preview,
            selection_status,
            source_info,
            frames_group,
            output_dir,
            process_status,
            process_preview,
        ],
    )

    preferred_preview.change(
        fn=apply_selected_preset,
        inputs=[preferred_preview, app_state],
        outputs=[app_state, selection_status, color_space, mask_channel],
    )

    frames_slider.change(
        fn=lambda value: slider_to_number(value, 1, MAX_FRAME_INPUT),
        inputs=frames_slider,
        outputs=frames_number,
    )
    frames_number.change(
        fn=lambda value: number_to_slider(value, 1, MAX_FRAME_INPUT),
        inputs=frames_number,
        outputs=[frames_slider, frames_number],
    )

    margin_slider.change(
        fn=lambda value: slider_to_number(value, 0, 255),
        inputs=margin_slider,
        outputs=margin_number,
    )
    margin_number.change(
        fn=lambda value: number_to_slider(value, 0, 255),
        inputs=margin_number,
        outputs=[margin_slider, margin_number],
    )

    kernel_slider.change(
        fn=lambda value: slider_to_number(value, 0, 31),
        inputs=kernel_slider,
        outputs=kernel_number,
    )
    kernel_number.change(
        fn=lambda value: number_to_slider(value, 0, 31),
        inputs=kernel_number,
        outputs=[kernel_slider, kernel_number],
    )

    dilate_slider.change(
        fn=lambda value: slider_to_number(value, 0, 31),
        inputs=dilate_slider,
        outputs=dilate_number,
    )
    dilate_number.change(
        fn=lambda value: number_to_slider(value, 0, 31),
        inputs=dilate_number,
        outputs=[dilate_slider, dilate_number],
    )

    blur_slider.change(fn=blur_slider_to_number, inputs=blur_slider, outputs=blur_number)
    blur_number.change(fn=blur_number_to_slider, inputs=blur_number, outputs=[blur_slider, blur_number])

    post_margin_slider.change(
        fn=lambda value: slider_to_number(value, 0, 255),
        inputs=post_margin_slider,
        outputs=post_margin_number,
    )
    post_margin_number.change(
        fn=lambda value: number_to_slider(value, 0, 255),
        inputs=post_margin_number,
        outputs=[post_margin_slider, post_margin_number],
    )

    process_button.click(
        fn=run_processing_step,
        inputs=[
            app_state,
            color_space,
            mask_channel,
            all_frames,
            frames_number,
            margin_number,
            kernel_number,
            dilate_number,
            blur_number,
            mask_out,
            post_process,
            post_margin_number,
            output_dir,
        ],
        outputs=[process_status, process_preview, output_dir],
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    ensure_directory(FIND_BEST_ROOT)
    ensure_directory(PROCESS_ROOT)
    demo.queue(default_concurrency_limit=1).launch()
