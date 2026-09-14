import json
from pathlib import Path
from tempfile import TemporaryDirectory
import streamlit as st
from src.adapters.mediapipe_pose import MediaPipeAdapter
from src.video import extract
from src.pose import validate_clip, animation_data
from src.animation import render

st.set_page_config(page_title='Pose Motion Lab · Python', layout='wide')
st.title('Pose Motion Lab · Python')
st.write('영상 → 공통 포즈 데이터 → 2D 애니메이션')
samples = Path(__file__).resolve().parents[2]/'sample-data'/'videos'
mode = st.radio('입력',['테스트 영상','영상 업로드','포즈 JSON'],horizontal=True)
source = None
pose_file = None
name = ''
if mode == '테스트 영상':
    options = sorted(p.relative_to(samples).as_posix() for p in samples.rglob('*.mp4'))
    if options:
        name = st.selectbox('영상 선택',options)
        source = (samples/name).read_bytes()
elif mode == '영상 업로드':
    upload = st.file_uploader('영상',type=['mp4','webm','avi'])
    if upload:
        source,name = upload.getvalue(),upload.name
else:
    pose_file = st.file_uploader('pose2d/v1 JSON',type=['json'])
fps = st.select_slider('추출 FPS',[10,15,30],value=15)
threshold = st.slider('표시 신뢰도',0.0,1.0,0.5,0.05)
st.caption('단일 인물 · CPU · 첫 추출에 모델 다운로드. 영상은 로컬 Python 서버에서 처리합니다.')
if st.button('추출 및 애니메이션 생성',disabled=source is None and pose_file is None):
    st.session_state.pop('result',None)
    try:
        with TemporaryDirectory(prefix='pose-demo-') as tmp:
            folder = Path(tmp)
            if pose_file is not None:
                clip = validate_clip(json.loads(pose_file.getvalue()))
            else:
                path = folder/('input'+Path(name).suffix)
                path.write_bytes(source)
                bar = st.progress(0.0,text='모델 준비 및 포즈 추출 중')
                clip = extract(path,MediaPipeAdapter(),fps,lambda v:bar.progress(v))
                clip['source']['name'] = name
            with st.spinner('애니메이션 생성 중'):
                preview = folder/'preview.mp4'
                render(clip,preview,threshold)
                st.session_state.result = dict(clip=clip,animation=animation_data(clip,threshold),preview=preview.read_bytes(),source=source,name=name or pose_file.name,threshold=threshold)
    except Exception as error:
        st.error(f'처리 실패: {error}')
if 'result' in st.session_state:
    r = st.session_state.result
    st.subheader(f"최근 결과 · {r['name']}")
    st.caption(f"{len(r['clip']['frames'])} frames · 신뢰도 {r['threshold']}. 설정 변경 후 다시 생성하세요.")
    left,right = st.columns(2)
    with left:
        st.write('원본 영상')
        if r['source']: st.video(r['source'])
        else: st.info('원본 영상 없이 JSON에서 생성했습니다.')
    with right:
        st.write('스켈레톤 애니메이션')
        st.video(r['preview'])
    st.download_button('포즈 JSON',json.dumps(r['clip'],allow_nan=False),'pose-motion.json','application/json')
    st.download_button('애니메이션 JSON',json.dumps(r['animation'],allow_nan=False),'pose-animation.json','application/json')
    st.download_button('미리보기 MP4',r['preview'],'pose-preview.mp4','video/mp4')
