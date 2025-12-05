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
CLASS_NAMES = ["Label 1", "Label 2", "Label 3","Label 4"] # あなたのラベルに書き換えて！

# =================================================
# Attention層
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

# MediaPipe設定
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils # ★描画用ツール

# ------------------------------------------------
# 映像処理クラス（デバッグフル装備）
# ------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.sequence = deque(maxlen=30)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.debug_text = "Initializing..."
        self.prob_text = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True

        # ★ 1. 骨格を画面に描画（これで見えてるか確認！）
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 検出フラグ
        has_pose = results.pose_landmarks is not None
        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None

        if model is not None:
            # --- 学習コードと同じ正規化処理 ---
            if has_pose:
                pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
            else:
                pose = np.zeros((33, 3))

            if has_lh:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
            else:
                lh = np.zeros((21, 3))
            
            if has_rh:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
            else:
                rh = np.zeros((21, 3))

            # 相対座標・正規化計算
            if np.sum(pose) != 0:
                left_shoulder = pose[11]
                right_shoulder = pose[12]
                center = (left_shoulder + right_shoulder) / 2.0
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                if shoulder_width < 0.01: shoulder_width = 1.0
            else:
                center = np.zeros(3)
                shoulder_width = 1.0

            pose_norm = (pose - center) / shoulder_width
            lh_norm = (lh - center) / shoulder_width
            rh_norm = (rh - center) / shoulder_width

            # 結合
            keypoints = np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])
            self.sequence.append(keypoints)

            # --- 予測 ---
            if len(self.sequence) == 30:
                input_data = np.expand_dims(list(self.sequence), axis=0)
                try:
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    confidence = prediction[0][predicted_index]

                    # ★ 閾値なしで生の数字を表示
                    label = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else str(predicted_index)
                    self.debug_text = f"Result: {label}"
                    self.prob_text = f"Conf: {confidence*100:.1f}%"
                    
                except Exception as e:
                    self.debug_text = "Error"
                    pass

        # ★ デバッグ情報の描画
        # センサー状況 P=Pose, L=Left, R=Right
        status = f"P[{'O' if has_pose else 'X'}] L[{'O' if has_lh else 'X'}] R[{'O' if has_rh else 'X'}]"
        
        # 黒い帯を引いて見やすくする
        cv2.rectangle(img, (0,0), (640, 90), (0, 0, 0), -1) 
        
        # 文字を書く
        cv2.putText(img, self.debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, self.prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, status, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img

st.title("🔍 完全デバッグモード")
st.write("体に緑の線が出ているか、P[O]になっているか確認してください")

webrtc_streamer(
    key="sign-language-debug-final",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
