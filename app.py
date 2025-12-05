import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K # ★追加：学習コードに合わせて追加
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
import av

# =================================================
# ⚙️ 設定エリア
# =================================================
MODEL_FILE_NAME = "best_sign_model.keras"

# ★あなたのクラス名に合わせて書き換えてください
CLASS_NAMES = ["Label 1", "Label 2", "Label 3", "動け！！"] 

# =================================================
# ★学習コードの「Attention」をそのまま移植
# =================================================
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x: (batch_size, time_steps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1) # 時間軸に対して重みを計算
        output = x * a
        return K.sum(output, axis=1) # 重み付き和を返す

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# =================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE_NAME, custom_objects={'Attention': Attention})

try:
    model = load_model()
    st.success(f"モデル読み込み成功！: {MODEL_FILE_NAME}")
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
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
            # ★ここが修正ポイント！学習コード「FEATURES = 225」に合わせます
            # 順番重要: Pose(33) -> Left Hand(21) -> Right Hand(21)

            # (1) Pose (33点 * 3 = 99次元)
            if results.pose_landmarks:
                pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
            else:
                pose = np.zeros(33*3)

            # (2) Left Hand (21点 * 3 = 63次元)
            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            else:
                lh = np.zeros(21*3)
            
            # (3) Right Hand (21点 * 3 = 63次元)
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            else:
                rh = np.zeros(21*3)

            # 全部つなげる (99 + 63 + 63 = 225次元！)
            keypoints = np.concatenate([pose, lh, rh])
            self.sequence.append(keypoints)

            # 3. 予測実行
            if len(self.sequence) == 30:
                input_data = np.expand_dims(list(self.sequence), axis=0)
                try:
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    confidence = prediction[0][predicted_index]

                    if confidence > 0.7: # 閾値
                        if predicted_index < len(CLASS_NAMES):
                            self.prediction_text = f"{CLASS_NAMES[predicted_index]} ({confidence*100:.1f}%)"
                        else:
                            self.prediction_text = f"Class {predicted_index}"
                except Exception as e:
                    # 次元が合わない等のエラーはここで無視されるので、今回はprintで出すようにしても良いかも
                    print(f"Prediction Error: {e}")
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
