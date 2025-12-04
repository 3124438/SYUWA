import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
import av

# ------------------------------------------------
# 1. 設定・モデル読み込み
# ------------------------------------------------
# ★重要★ ラベルを学習させた順番・内容に合わせて書き換えてください
LABELS = ['Label 1', 'Label 2', 'Label 3'] 

@st.cache_resource
def load_model():
    # モデルのパスを指定
    return tf.keras.models.load_model('sign_language_model.h5')

try:
    model = load_model()
    st.success("モデル読み込み成功！")
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
    model = None

# MediaPipe設定
mp_holistic = mp.solutions.holistic

# ------------------------------------------------
# 2. 映像処理クラス
# ------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # 30フレーム分のデータを貯める箱
        self.sequence = deque(maxlen=30)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prediction_text = "Waiting..."

    def transform(self, frame):
        # WebRTCから画像を取得
        img = frame.to_ndarray(format="bgr24")

        # 1. 骨格抽出 (MediaPipe)
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True

        # 2. 骨格を画面に描画（確認用：重ければコメントアウトしてください）
        # mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp.solutions.drawing_utils.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 3. 座標データ変換
        if model is not None:
            # 左手・右手の検出
            # ★注意★ 学習時のデータ処理と全く同じにする必要があります
            if results.left_hand_landmarks:
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            else:
                lh = np.zeros(21*3)
            
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            else:
                rh = np.zeros(21*3)

            # データ結合してキューに追加 (左手+右手 = 126次元)
            keypoints = np.concatenate([lh, rh])
            self.sequence.append(keypoints)

            # 4. 30フレーム溜まったら予測を実行
            if len(self.sequence) == 30:
                # リアルタイム性を保つため、データを整える
                input_data = np.expand_dims(list(self.sequence), axis=0)
                
                try:
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    confidence = prediction[0][predicted_index]

                    # 信頼度が70%以上のときだけ表示更新
                    if confidence > 0.7:
                        # 範囲外エラーを防ぐ
                        if predicted_index < len(LABELS):
                            self.prediction_text = f"{LABELS[predicted_index]} ({confidence*100:.1f}%)"
                        else:
                            self.prediction_text = f"Class {predicted_index}"
                except Exception as e:
                    print(f"Prediction Error: {e}")

        # 5. 結果を画面に書き込む
        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, self.prediction_text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img

# ------------------------------------------------
# 3. アプリ画面構成
# ------------------------------------------------
st.title("🤟 リアルタイム手話判定")
st.write("カメラを許可して、手を動かしてください（30フレーム蓄積後に判定します）")

# WebRTCの起動設定
webrtc_streamer(
    key="sign-language",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
