"""Only this adapter knows MediaPipe indices. estimate accepts RGB + milliseconds."""
from pathlib import Path
from urllib.request import urlopen
import os
import tempfile
import mediapipe as mp
from src.pose import JOINTS

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
MODEL = Path(__file__).resolve().parents[2] / '.cache' / 'pose_landmarker_lite.task'
INDICES = [0,11,12,13,14,15,16,23,24,25,26,27,28]

class MediaPipeAdapter:
    def initialize(self):
        if not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=MODEL.parent, delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    with urlopen(MODEL_URL, timeout=60) as response:
                        temp.write(response.read())
                except Exception:
                    temp.close()
                    temporary.unlink(missing_ok=True)
                    raise
            os.replace(temporary, MODEL)
        self.model = mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1))

    def estimate(self, rgb, timestamp_ms):
        result = self.model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),timestamp_ms)
        pose = result.pose_landmarks[0] if result.pose_landmarks else None
        return {name: {'x':float(pose[i].x),'y':float(pose[i].y),'confidence':float(pose[i].visibility)} if pose else None for name,i in zip(JOINTS,INDICES)}

    def close(self):
        if hasattr(self, 'model'):
            self.model.close()
