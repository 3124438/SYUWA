import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
import av

# =================================================
# ⚙️ 設定エリア
# =================================================
MODEL_FILE_NAME = "best_sign_model.keras"
CLASS_NAMES = ["Label 1", "Label 2", "Label 3","動け！！！"] # ★あなたのクラス名に合わせてね！

# =================================================
# Attention層 (変更なし)
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
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE_NAME, custom_objects={'Attention': Attention})

try:
    model = load_model()
    st.success(f"モデル読み込み成功: {MODEL_FILE_NAME}")
except Exception as e:
    st.error(f"エラー: {e}")
    model = None

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

        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True

        if model is not None:
            # ---------------------------------------------------------
            # ★ここが劇的変化！学習コードと同じ「計算（正規化）」をします
            # ---------------------------------------------------------
            
            # 1. 生データを取得（なければゼロ埋め）
            if results.pose_landmarks:
                pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
            else:
                pose = np.zeros((33, 3))

            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
            else:
                lh = np.zeros((21, 3))
            
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
            else:
                rh = np.zeros((21, 3))

            # 2. 相対座標へ変換（process_landmarks_relative のロジック）
            # ポーズデータがある場合のみ計算可能
            if np.sum(pose) != 0:
                # 肩（11番と12番）の中点を計算
                left_shoulder = pose[11]
                right_shoulder = pose[12]
                center = (left_shoulder + right_shoulder) / 2.0
                
                # 肩幅を計算（これを基準の「1」とする）
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                if shoulder_width < 0.01: shoulder_width = 1.0 # ゼロ除算防止
            else:
                center = np.zeros(3)
                shoulder_width = 1.0

            # 3. 正規化（中心を引いて、肩幅で割る）
            pose_norm = (pose - center) / shoulder_width
            lh_norm = (lh - center) / shoulder_width
            rh_norm = (rh - center) / shoulder_width

            # 4. 一列に並べる (33*3 + 21*3 + 21*3 = 225次元)
            keypoints = np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])
            
            # キューに追加
            self.sequence.append(keypoints)

            # 5. 予測実行 (30フレーム溜まったら)
            if len(self.sequence) == 30:
                input_data = np.expand_dims(list(self.sequence), axis=0)
                try:
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    confidence = prediction[0][predicted_index]

                    # 判定
                    label = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else str(predicted_index)
                    
                    if confidence > 0.8: # 自信があるときだけ更新
                        self.prediction_text = f"{label} ({confidence*100:.1f}%)"
                    
                except Exception as e:
                    pass

        # 描画
        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, self.prediction_text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img

st.title("🤟 手話認識アプリ（正規化対応版）")
st.write(f"Model: {MODEL_FILE_NAME}")

webrtc_streamer(
    key="sign-language-norm",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
