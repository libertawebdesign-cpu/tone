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

# トップページ ( / ) にアクセスしたときに index.html を返す
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# 画像解析のAPI ( /api/preview )
@app.post("/api/preview")
async def generate_preview(ref: UploadFile = File(...), target: UploadFile = File(...)):
    ref_bytes = await ref.read()
    target_bytes = await target.read()
    
    # 画像をメモリ上で読み込む
    ref_img = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
    target_img = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)

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

    # JPEG画像としてエンコードして返す
    _, encoded_img = cv2.imencode('.jpg', result_bgr)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")