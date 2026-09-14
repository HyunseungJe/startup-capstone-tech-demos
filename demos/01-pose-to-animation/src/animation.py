import math
import cv2
import numpy as np
import imageio_ffmpeg
from src.pose import sample, bone_transforms

def render(clip, path, threshold=.5, fps=30):
    source = clip['source']
    scale = 640/max(source['width'],source['height'])
    w,h = [max(2,round(source[k]*scale/2)*2) for k in ('width','height')]
    writer = imageio_ffmpeg.write_frames(str(path),(w,h),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,output_params=['-movflags','+faststart'])
    writer.send(None)
    try:
        for i in range(math.ceil(source['duration']*fps)):
            points = sample(clip,i/fps)
            image = np.full((h,w,3), (14,21,32), dtype=np.uint8)
            for x in range(0,w,32): cv2.line(image,(x,0),(x,h),(26,38,54),1)
            for y in range(0,h,32): cv2.line(image,(0,y),(w,y),(26,38,54),1)
            for b in bone_transforms(points,w,h,threshold):
                start = (round(b['x']),round(b['y']))
                end = (round(b['x']+math.cos(b['rotation'])*b['length']),round(b['y']+math.sin(b['rotation'])*b['length']))
                cv2.line(image,start,end,(165,233,201) if b['from'].startswith('left') else (143,174,255),4,cv2.LINE_AA)
            for p in points.values():
                if p and p['confidence'] >= threshold:
                    cv2.circle(image,(round(p['x']*w),round(p['y']*h)),4,(233,244,255),-1,cv2.LINE_AA)
            writer.send(image)
    finally:
        writer.close()
