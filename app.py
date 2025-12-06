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
# クラス名は4つ（あなたの環境に合わせて！）
CLASS_NAMES = ["Label 1", "Label 2", "Label 3", "動け！！！"] 

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
except Exception as e:
    st.error(f"エラー: {e}")
    model = None

# MediaPipe設定
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =================================================
# 🎛️ UIサイドバー設定
# =================================================
st.sidebar.title("設定パネル")
# デバッグモードの切り替えスイッチ
DEBUG_MODE = st.sidebar.checkbox("デバッグモード（骨格表示）", value=True)
st.sidebar.write("---")
st.sidebar.write("映像の右側にリアルタイム分析結果が表示されます。")

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
        # 初期値
        self.probs = np.zeros(len(CLASS_NAMES)) 
        self.result_label = "Waiting..."
        self.result_conf = 0.0
        self.status_text = "Init..."
        
        # UIから受け取った設定（クラス作成時に渡せないので、global変数を参照する形をとります）
        self.debug = DEBUG_MODE

    def transform(self, frame):
        # 1. 画像の準備
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # ★ ダッシュボード用のキャンバスを作成（横幅を広げる）
        # 元の画像(w) + 右側のパネル(300px)
        panel_w = 320
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        
        # 左側にカメラ映像をコピー
        canvas[:h, :w] = img

        # MediaPipe処理（左側の画像に対して行う）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # ---------------------------------------------------------
        # 2. 骨格描画（デバッグモードONの時だけ）
        # ---------------------------------------------------------
        if self.debug:
            # キャンバスの左側(カメラ部分)に描画
            camera_area = canvas[:h, :w]
            mp_drawing.draw_landmarks(camera_area, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(camera_area, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(camera_area, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # ---------------------------------------------------------
        # 3. データ抽出と予測
        # ---------------------------------------------------------
        has_pose = results.pose_landmarks is not None
        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None
        
        self.status_text = f"P[{'O' if has_pose else 'X'}] L[{'O' if has_lh else 'X'}] R[{'O' if has_rh else 'X'}]"

        if model is not None:
            # 正規化ロジック
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

            keypoints = np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])
            self.sequence.append(keypoints)

            if len(self.sequence) == 30:
                input_data = np.expand_dims(list(self.sequence), axis=0)
                try:
                    prediction = model.predict(input_data, verbose=0)
                    self.probs = prediction[0] # 全確率を保存
                    idx = np.argmax(self.probs)
                    self.result_conf = self.probs[idx]
                    
                    if idx < len(CLASS_NAMES):
                        self.result_label = CLASS_NAMES[idx]
                    else:
                        self.result_label = f"Class {idx}"

                except Exception:
                    pass

        # ---------------------------------------------------------
        # 4. ダッシュボード描画（右側の黒い部分）
        # ---------------------------------------------------------
        # 基準位置 (右側のパネルの開始位置)
        x_start = w + 10
        y_cursor = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # タイトルエリア
        cv2.putText(canvas, "AI Analysis", (x_start, y_cursor), font, 0.8, (255, 255, 255), 2)
        y_cursor += 40
        
        # センサー状況
        # 色を変える (OKなら緑、NGなら赤)
        p_color = (0, 255, 0) if has_pose else (0, 0, 255)
        cv2.putText(canvas, self.status_text, (x_start, y_cursor), font, 0.5, p_color, 1)
        y_cursor += 40
        
        # 区切り線
        cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
        y_cursor += 30

        # 結果表示 (大きく)
        cv2.putText(canvas, "Result:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
        y_cursor += 35
        # 結果ラベル（黄色）
        cv2.putText(canvas, self.result_label, (x_start, y_cursor), font, 1.0, (0, 255, 255), 2)
        y_cursor += 30
        # 信頼度
        cv2.putText(canvas, f"Conf: {self.result_conf*100:.1f}%", (x_start, y_cursor), font, 0.6, (0, 255, 255), 1)
        
        y_cursor += 40
        cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
        y_cursor += 30

        # グラフ描画エリア
        cv2.putText(canvas, "Probabilities:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
        y_cursor += 20

        # 各クラスのバーグラフ
        bar_max_width = 180 # バーの最大長さ
        for i, prob in enumerate(self.probs):
            class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            
            # クラス名
            y_cursor += 20
            cv2.putText(canvas, f"{class_name}", (x_start, y_cursor), font, 0.5, (255, 255, 255), 1)
            
            # バーの背景（グレー）
            y_bar = y_cursor + 5
            cv2.rectangle(canvas, (x_start, y_bar), (x_start + bar_max_width, y_bar + 10), (50, 50, 50), -1)
            
            # バーの中身（確率に応じて長さ可変、色は青）
            bar_w = int(prob * bar_max_width)
            # 予測トップなら色を赤にする、それ以外は緑
            bar_color = (0, 0, 255) if prob == max(self.probs) else (0, 255, 0)
            
            if bar_w > 0:
                cv2.rectangle(canvas, (x_start, y_bar), (x_start + bar_w, y_bar + 10), bar_color, -1)
            
            # パーセント数値
            cv2.putText(canvas, f"{prob*100:.0f}%", (x_start + bar_max_width + 10, y_bar + 8), font, 0.4, (200, 200, 200), 1)
            y_cursor += 20 # 次の行へ

        return canvas

# ------------------------------------------------
# アプリ画面構成
# ------------------------------------------------
st.title("🤟 AI手話解析システム")

if model is None:
    st.error("モデルが読み込めませんでした。")
else:
    # WebRTCの起動
    webrtc_streamer(
        key="sign-language-dashboard",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
