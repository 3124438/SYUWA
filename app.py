import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import av

# 1. モデルの読み込み
# あなたがアップロードしたファイル名と一致させています
MODEL_PATH = "sign_language_model.h5"

# クラス名（必要に応じて書き換えてください）
CLASS_NAMES = ["Label 1", "Label 2", "Label 3"] 

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("モデル読み込み成功！")
except Exception as e:
    st.error(f"エラー: {e}")
    model = None

# 2. 映像処理
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if model is not None:
            # 前処理 (224x224にリサイズなど。学習時の設定に合わせてください)
            img_resized = cv2.resize(img, (224, 224))
            img_input = img_resized.astype('float32') / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            # 推論
            prediction = model.predict(img_input)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]
            result_text = CLASS_NAMES[predicted_index]

            # 結果表示
            cv2.putText(img, f"{result_text} ({confidence:.2f})", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img

# 3. アプリ画面
st.title("手話認識アプリ")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
