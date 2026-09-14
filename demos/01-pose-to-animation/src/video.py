import math
from pathlib import Path
import cv2
from src.pose import create_clip

def extract(path, adapter, fps=15, progress=lambda value: None):
    cap = cv2.VideoCapture(str(path))
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(source_fps) or source_fps <= 0 or total <= 0:
            raise ValueError('영상을 읽을 수 없습니다. MP4 또는 WebM을 사용하세요.')
        rate = min(fps, source_fps)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, index, next_t = [], 0, 0
        adapter.initialize()
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = index/source_fps
            if t + 1e-8 >= next_t:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append({'t':t,'joints':adapter.estimate(rgb, round(t*1000))})
                next_t += 1/rate
                progress(min(1, (index+1)/total))
            index += 1
        if index < total-1:
            raise ValueError(f'영상 디코딩이 중단되었습니다: {index}/{total} 프레임')
        progress(1.0)
        return create_clip(dict(name=Path(path).name,width=width,height=height,duration=index/source_fps,adapter=type(adapter).__name__),rate,frames)
    finally:
        cap.release()
        adapter.close()
