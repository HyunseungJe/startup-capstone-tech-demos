"""Model-independent pose2d/v1 representation, compatible with demo 01."""
import math
from bisect import bisect_right

JOINTS = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
BONES = [['left_shoulder','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle']]
COORDINATES = 'normalized-image-x-right-y-down'

def create_clip(source, fps, frames):
    return validate_clip(dict(schema='pose2d/v1', coordinates=COORDINATES, source=source, fps=fps, joints=JOINTS, bones=BONES, frames=frames))

def validate_clip(c):
    def finite(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not isinstance(c, dict) or c.get('schema') != 'pose2d/v1' or c.get('coordinates') != COORDINATES:
        raise ValueError('pose2d/v1 JSON이 필요합니다.')
    source = c.get('source', {})
    if any(not finite(source.get(k)) or source[k] <= 0 for k in ('width','height','duration')):
        raise ValueError('영상 크기와 길이가 잘못되었습니다.')
    if not c.get('frames') or not finite(c.get('fps')) or c['fps'] <= 0:
        raise ValueError('프레임 또는 FPS가 없습니다.')
    prev = -1
    for f in c['frames']:
        if not finite(f.get('t')) or not prev < f['t'] <= source['duration'] or f['t'] < 0 or not isinstance(f.get('joints'), dict):
            raise ValueError('프레임 시간이 잘못되었습니다.')
        prev = f['t']
        for p in f['joints'].values():
            if p is not None and (not isinstance(p, dict) or any(not finite(p.get(k)) for k in ('x','y','confidence')) or not 0 <= p['confidence'] <= 1):
                raise ValueError('관절 데이터가 잘못되었습니다.')
    return c

def sample(clip, t):
    frames = clip['frames']
    i = max(0, bisect_right([f['t'] for f in frames], t)-1)
    a, b = frames[i], frames[min(i+1, len(frames)-1)]
    u = max(0, min(1, (t-a['t'])/(b['t']-a['t']))) if b['t'] != a['t'] else 0
    result = {}
    for key in JOINTS:
        p, q = a['joints'].get(key), b['joints'].get(key)
        result[key] = ({'x':p['x']+(q['x']-p['x'])*u, 'y':p['y']+(q['y']-p['y'])*u, 'confidence':min(p['confidence'],q['confidence'])} if p and q else p if u == 0 else None)
    return result

def bone_transforms(points, width, height, threshold=.5):
    result = []
    for start, end in BONES:
        a, b = points.get(start), points.get(end)
        if not a or not b or min(a['confidence'], b['confidence']) < threshold:
            continue
        dx, dy = (b['x']-a['x'])*width, (b['y']-a['y'])*height
        result.append(dict(zip(('from','to','x','y','rotation','length'), (start,end,a['x']*width,a['y']*height,math.atan2(dy,dx),math.hypot(dx,dy)))))
    return result

def animation_data(clip, threshold=.5):
    s = clip['source']
    return {'schema':'pose2d-animation/v1','duration':s['duration'],'space':dict(width=s['width'],height=s['height'],units='source-pixels',rotation='radians-clockwise-from-positive-x'),'confidenceThreshold':threshold,'frames':[{'t':f['t'],'bones':bone_transforms(f['joints'],s['width'],s['height'],threshold)} for f in clip['frames']]}
