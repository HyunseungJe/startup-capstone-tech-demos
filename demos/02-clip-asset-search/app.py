"""Small, local Jina CLIP v2 asset search demo. Run with uv run python app.py."""

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
from transformers import AutoModel


MODEL_ID = "jinaai/jina-clip-v2"
EMBEDDING_DIM = 512
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
MODEL_LOCK = Lock()
MODEL = None


def runtime_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            MODEL = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(runtime_device()).eval()
    return MODEL


def asset_root(folder: str) -> Path:
    if not folder.strip():
        raise ValueError("에셋 폴더 경로를 입력하세요.")
    root = Path(folder.strip().strip('"')).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {root}")
    return root


def cache_path(root: Path) -> Path:
    key = hashlib.sha256(
        f"{root}|{MODEL_ID}|dim-{EMBEDDING_DIM}|rgb-background-240-v1".encode()
    ).hexdigest()
    return CACHE_DIR / f"{key}.npz"


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (240, 240, 240, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def normalized(values) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def encode_images(model, images) -> np.ndarray:
    try:
        embeddings = model.encode_image(images, truncate_dim=EMBEDDING_DIM)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU 메모리가 부족합니다. 인덱싱 배치 크기를 줄이거나 CPU로 실행하세요."
        ) from error
    return normalized(embeddings)


def encode_query(model, query: str) -> np.ndarray:
    embeddings = model.encode_text(
        [query], task="retrieval.query", truncate_dim=EMBEDDING_DIM
    )
    return normalized(embeddings)[0]


def build_index(folder: str, progress=None) -> str:
    root = asset_root(folder)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError("PNG, JPG, JPEG, WebP 이미지가 없습니다.")
    if progress:
        progress(0, desc=f"Jina CLIP v2 로딩 중 ({runtime_device()})")
    model = get_model()
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
            with torch.inference_mode():
                features.append(encode_images(model, images))
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
        raise ValueError("검색 문장을 입력하세요.")
    target = cache_path(root)
    if not target.exists():
        raise ValueError("이 폴더의 인덱스가 없습니다. 먼저 인덱싱하세요.")
    with np.load(target, allow_pickle=False) as index:
        paths = index["paths"].tolist()
        vectors = index["vectors"]
    model = get_model()
    with torch.inference_mode():
        query_vector = encode_query(model, query.strip())
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
    device = runtime_device()
    with gr.Blocks(title="Jina CLIP v2 Game Asset Search") as demo:
        gr.Markdown(
            "# Jina CLIP v2 Game Asset Search\n"
            "파일명이나 태그 없이 한국어·영어 문장으로 게임 에셋을 검색합니다.\n\n"
            f"`{MODEL_ID}` · 512차원 임베딩 · 실행 장치: **{device}**"
        )
        folder = gr.Textbox(label="로컬 에셋 폴더", placeholder=r"C:\assets\icons")
        index_button = gr.Button("인덱싱 / 다시 인덱싱")
        index_status = gr.Textbox(label="인덱스 상태", interactive=False)
        gr.Markdown("최초 인덱싱 시 약 0.9B 파라미터 모델을 다운로드합니다. 저장된 인덱스는 재실행 후에도 사용할 수 있습니다. 에셋 추가·수정·삭제 후에는 다시 인덱싱하세요.")
        query = gr.Textbox(label="검색 문장 (한국어 또는 영어)", placeholder="붉은색 회복 포션")
        gr.Examples(
            examples=[
                ["붉은색 회복 포션"],
                ["근육질 판타지 전사"],
                ["어두운 숲 배경"],
                ["a rusty sword"],
                ["a blue magic item"],
            ],
            inputs=query,
        )
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
