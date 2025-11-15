from model import VideoPipeline
from keras import applications
from tensorflow import keras
import streamlit as st
import numpy as np
import tempfile, torch, os

def preprocess_resnet(x):
    return applications.resnet50.preprocess_input(x)

@st.cache_resource
def load_model():
    return keras.models.load_model(
        # You can use any car crash detection model here
        '../modele.keras',
        custom_objects={
            'preprocess_resnet': preprocess_resnet
        }
    )


st.title('🚗 Real-Time Car Crash Detection ')
st.write('Upload a video, then the pipeline and the model prediction will run.')

uploaded_file = st.file_uploader('Upload MP4 video', type=['mp4'])

if uploaded_file:
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(video_path, 'wb') as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    pre_status = st.empty()
    pre_status.info('Processing video… Extracting frames, optical-flow divergence mask…')
    pipeline = VideoPipeline(video_path)

    try:
        video_np_path, mask_np_path = pipeline.process()

        pre_status.empty()

        target = pipeline.get_target()
        target = '🚨 CRASH' if target == 1 else '✅ SAFE'

        st.success('Video processing completed!')

        video_np = np.load(video_np_path)
        mask_np = np.load(mask_np_path)

        st.subheader('🔍 Sample processed frame')
        st.image(video_np[0].astype(np.uint8), caption='Resized video frame (224×224)')

        st.subheader('🔍 Sample mask frame')
        mask_sample = mask_np[0].transpose(1, 2, 0)
        st.image(mask_sample, caption='Optical-flow mask')

        status = st.empty()
        status.info('Running model prediction…')
        model = load_model()

        video_input = np.expand_dims(video_np, axis=0).astype('float32')
        mask_input = np.expand_dims(mask_np, axis=0).astype('float32')

        pred = model.predict([video_input, mask_input])[0][0]
        pred_label = '🚨 CRASH' if pred > 0.5 else '✅ SAFE'

        status.empty()

        st.success('Model Prediction completed!')

        st.subheader('🧠 Model Prediction')
        st.metric('Prediction', pred_label, f'score={pred*100:.3f}%')
        st.metric('The Real Label', target)

    except Exception as e:
        st.error(f'Error during processing: {e}')
