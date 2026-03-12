from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
import cv2
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")

# 🌟 追加：メモリ不足（OOM）対策用のリサイズ関数
def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    # 縦か横、長い方が max_size を超えていたら縮小する
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

@app.post("/api/preview")
async def generate_preview(ref: UploadFile = File(...), target: UploadFile = File(...)):
    ref_bytes = await ref.read()
    target_bytes = await target.read()
    
    # 画像を読み込む
    ref_img = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
    target_img = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 🌟 追加：ここで画像をリサイズしてサーバーのパンクを防ぐ！
    ref_img = resize_image(ref_img)
    target_img = resize_image(target_img)

    # LAB色空間に変換
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")

    # 色の平均とばらつきを計算
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)
    
    # ゼロ割り算エラーの回避
    target_std = np.where(target_std == 0, 1e-6, target_std)

    # Reinhard法でカラーマッチング
    shifted_lab = ((target_lab - target_mean.flatten()) * (ref_std.flatten() / target_std.flatten())) + ref_mean.flatten()
    shifted_lab = np.clip(shifted_lab, 0, 255).astype("uint8")
    
    # BGRに戻す
    result_bgr = cv2.cvtColor(shifted_lab, cv2.COLOR_LAB2BGR)

    # JPEG画像としてエンコードして返す（画質を85%にしてさらに軽量化）
    _, encoded_img = cv2.imencode('.jpg', result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
