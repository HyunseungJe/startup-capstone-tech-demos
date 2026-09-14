import json
from pathlib import Path
import cv2
from src.video import extract
from src.adapters.mediapipe_pose import MediaPipeAdapter
from src.animation import render
from src.pose import animation_data

if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    out = base/'outputs'
    out.mkdir(exist_ok=True)
    for name in ['exercise/walk_hd.mp4','exercise/squat.mp4','hmdb51/sword_01.mp4']:
        path = base.parents[1]/'sample-data'/'videos'/name
        clip = extract(path,MediaPipeAdapter(),10)
        prefix = out/path.stem
        prefix.with_suffix('.pose.json').write_text(json.dumps(clip,allow_nan=False),encoding='utf-8')
        prefix.with_suffix('.animation.json').write_text(json.dumps(animation_data(clip),allow_nan=False),encoding='utf-8')
        preview = prefix.with_suffix('.mp4')
        render(clip,preview)
        cap = cv2.VideoCapture(str(preview))
        ok,_ = cap.read()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert ok and count > 0
        detected = sum(any(p and p['confidence'] >= .5 for p in f['joints'].values()) for f in clip['frames'])
        print(f'{name}: {len(clip["frames"])} pose frames, {detected} detected, {count} preview frames',flush=True)
