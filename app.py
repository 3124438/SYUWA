import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Layer # ★追加
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
import av

# =================================================
# ⚙️ 設定エリア
# =================================================
MODEL_FILE_NAME = "best_sign_model.keras"
CLASS_NAMES = ["Label 1", "Label 2", "Label 3", "動け!!"] 

# =================================================
# ★ここに「Attention」の設計図を追加しました！
# =================================================
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='normal', 
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros', 
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# =================================================

@st.cache_resource
def load_model():
    # ★ここも修正！「Attentionを使ってね」と教えています
    return tf.keras.models.load_model(MODEL_FILE_NAME, custom_objects={'Attention': Attention})

# モデル読み込み処理
try:
    model = load_model()
    st.success(f"モデル読み込み成功！: {MODEL_FILE_NAME}")
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
    st.error("※学習コードのAttentionと形が違う可能性があります。その場合は学習コードを見せてください！")
    model = None

# MediaPipe設定
mp_holistic = mp.solutions.holistic

# ------------------------------------------------
# 映像処理クラス
# ------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.sequence = deque(maxlen=30)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prediction_text = "Waiting..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 1. 骨格抽出
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True

        # 2. 座標データ変換
        if model is not None:
            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            else:
                lh = np.zeros(21*3)
            
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            else:
                rh = np.zeros(21*3)

            keypoints = np.concatenate([lh, rh])
            self.sequence.append(keypoints)

            # 3. 予測実行
            if len(self.sequence) == 30:
                input_data = np.expand_dims(list(self.sequence), axis=0)
                try:
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    confidence = prediction[0][predicted_index]

                    if confidence > 0.7:
                        if predicted_index < len(CLASS_NAMES):
                            self.prediction_text = f"{CLASS_NAMES[predicted_index]} ({confidence*100:.1f}%)"
                        else:
                            self.prediction_text = f"Class {predicted_index}"
                except Exception as e:
                    pass

        # 4. 描画
        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, self.prediction_text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img

# ------------------------------------------------
# アプリ画面構成
# ------------------------------------------------
st.title("🤟 手話リアルタイム認識")
st.write(f"読み込みモデル: {MODEL_FILE_NAME}")

webrtc_streamer(
    key="sign-language",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
