"""Small, local CLIP asset search demo. Run with uv run python app.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

import gradio as gr
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
MODEL_LOCK = Lock()
MODEL = None
PROCESSOR = None


def get_model():
    global MODEL, PROCESSOR
    with MODEL_LOCK:
        if MODEL is None:
            processor = CLIPProcessor.from_pretrained(MODEL_ID)
            model = CLIPModel.from_pretrained(MODEL_ID).eval()
            PROCESSOR, MODEL = processor, model
    return MODEL, PROCESSOR


def asset_root(folder: str) -> Path:
    if not folder.strip():
        raise ValueError("에셋 폴더 경로를 입력하세요.")
    root = Path(folder.strip().strip('"')).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {root}")
    return root


def cache_path(root: Path) -> Path:
    key = hashlib.sha256(f"{root}|{MODEL_ID}|rgb-background-240-v1".encode()).hexdigest()
    return CACHE_DIR / f"{key}.npz"


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (240, 240, 240, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def normalized(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return tensor.detach().cpu().numpy().astype(np.float32)


def build_index(folder: str, progress=None) -> str:
    root = asset_root(folder)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError("PNG, JPG, JPEG, WebP 이미지가 없습니다.")
    if progress:
        progress(0, desc="CLIP 로딩 중 (최초 실행 시 모델 다운로드)")
    model, processor = get_model()
    names, features, skipped = [], [], []
    batch_size = 16
    for start in range(0, len(files), batch_size):
        images, batch_names = [], []
        for path in files[start:start + batch_size]:
            try:
                images.append(read_image(path))
                batch_names.append(str(path.relative_to(root)))
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                skipped.append(str(path.relative_to(root)))
        if images:
            inputs = processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                features.append(normalized(model.get_image_features(**inputs)))
            names.extend(batch_names)
        if progress:
            progress(min(start + batch_size, len(files)) / len(files), desc=f"인덱싱: {min(start + batch_size, len(files))}/{len(files)}")
    if not features:
        raise ValueError("읽을 수 있는 이미지가 없습니다. 기존 캐시는 유지됩니다.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Replace only after the entire index has been built successfully.
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, paths=np.asarray(names), vectors=np.concatenate(features))
    try:
        os.replace(temporary, cache_path(root))
    finally:
        temporary.unlink(missing_ok=True)
    message = f"인덱싱 완료: {len(names)}개 이미지 · 읽기 실패 {len(skipped)}개"
    if skipped:
        message += "\n건너뛴 파일 (최대 10개): " + ", ".join(skipped[:10])
    return message


def search_assets(folder: str, query: str, top_k: int = 12):
    root = asset_root(folder)
    if not query.strip():
        raise ValueError("영어 검색 문장을 입력하세요.")
    target = cache_path(root)
    if not target.exists():
        raise ValueError("이 폴더의 인덱스가 없습니다. 먼저 인덱싱하세요.")
    with np.load(target, allow_pickle=False) as index:
        paths = index["paths"].tolist()
        vectors = index["vectors"]
    model, processor = get_model()
    inputs = processor(text=[query.strip()], return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        query_vector = normalized(model.get_text_features(**inputs))[0]
    scores = vectors @ query_vector
    results = []
    for position in np.argsort(-scores):
        relative = paths[int(position)]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        results.append({"path": str(path), "name": relative, "score": float(scores[position])})
        if len(results) >= top_k:
            break
    return results


def index_ui(folder, progress=gr.Progress()):
    try:
        return build_index(folder, progress)
    except Exception as error:
        raise gr.Error(f"인덱싱 실패: {error}") from error


def search_ui(folder, query, top_k):
    try:
        results = search_assets(folder, query, int(top_k))
        gallery = []
        skipped = 0
        for result in results:
            try:
                thumbnail = read_image(Path(result["path"]))
                thumbnail.thumbnail((384, 384))
                gallery.append((thumbnail, f"{result['name']} · {result['score']:.3f}"))
            except (OSError, ValueError, Image.DecompressionBombError):
                skipped += 1
        status = f"{len(gallery)}개 결과 · 점수는 코사인 유사도이며 확률이 아닙니다."
        if skipped or not gallery:
            status += " 읽을 수 없는 파일이 있으면 다시 인덱싱하세요."
        return gallery, status
    except Exception as error:
        raise gr.Error(f"검색 실패: {error}") from error


def create_app():
    with gr.Blocks(title="CLIP Game Asset Search") as demo:
        gr.Markdown("# CLIP Game Asset Search\n파일명이나 태그 없이, 이미지의 의미로 게임 에셋을 검색합니다.")
        folder = gr.Textbox(label="로컬 에셋 폴더", placeholder=r"C:\assets\icons")
        index_button = gr.Button("인덱싱 / 다시 인덱싱")
        index_status = gr.Textbox(label="인덱스 상태", interactive=False)
        gr.Markdown("최초 인덱싱 시 모델을 다운로드합니다. 저장된 인덱스는 재실행 후에도 사용할 수 있습니다. 에셋 추가·수정·삭제 후에는 다시 인덱싱하세요.")
        query = gr.Textbox(label="검색 문장 (영어)", placeholder="a rusty sword")
        gr.Examples(examples=[["a rusty sword"], ["a red health potion"], ["a wooden treasure chest"], ["a round metal shield"]], inputs=query)
        top_k = gr.Slider(minimum=1, maximum=24, value=12, step=1, label="결과 개수")
        search_button = gr.Button("검색", variant="primary")
        search_status = gr.Textbox(label="검색 상태", interactive=False)
        gallery = gr.Gallery(label="검색 결과", columns=4, object_fit="contain", height=600)
        index_button.click(index_ui, inputs=folder, outputs=index_status, concurrency_id="clip", concurrency_limit=1)
        search_button.click(search_ui, inputs=[folder, query, top_k], outputs=[gallery, search_status], concurrency_id="clip", concurrency_limit=1)
        query.submit(search_ui, inputs=[folder, query, top_k], outputs=[gallery, search_status], concurrency_id="clip", concurrency_limit=1)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", metavar="FOLDER", help="Build an index without opening the UI")
    parser.add_argument("--search", metavar="QUERY", help="Search the folder specified by --folder")
    parser.add_argument("--folder", help="Asset folder for command-line search")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.index:
        print(build_index(args.index))
    elif args.search:
        if not args.folder:
            parser.error("--search requires --folder")
        print(json.dumps(search_assets(args.folder, args.search), ensure_ascii=False, indent=2))
    else:
        create_app().queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)
