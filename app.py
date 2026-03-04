# -*- coding: utf-8 -*-
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

# 1. 加载模型
MODEL_PATH = 'invasive_plant_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# 2. 类别名称（严格对应dataset顺序）
class_names = ['水葫芦', '紫茎泽兰', '鬼针草']

# 3. 预处理函数（和训练一致）
def preprocess_image(image_file):
    img = Image.open(io.BytesIO(image_file)).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 4. 主页：直接把HTML写在这里，彻底解决编码问题
@app.get("/", response_class=HTMLResponse)
async def main():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>入侵植物识别</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; text-align: center; }
        h1 { color: #2c3e50; }
        #result { margin-top: 20px; padding: 20px; background: #f0f8ff; border-radius: 10px; }
        button { background: #3498db; color: white; border: none; padding: 10px 25px; border-radius: 5px; cursor: pointer; margin-top: 15px; }
        input { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🌿 入侵植物识别系统</h1>
    <input type="file" id="fileInput" accept="image/*" capture="camera">
    <br>
    <button onclick="predictImage()">开始识别</button>
    <div id="result"></div>

    <script>
        async function predictImage() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('请先选择图片！'); return; }
            
            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('result').innerHTML = 
                    '<strong>预测结果：</strong>' + data.predicted_class + '<br><br>' +
                    '<strong>置信度：</strong>' + data.confidence;
            } catch (e) {
                document.getElementById('result').innerHTML = '<span style="color:red;">识别失败，请重试！</span>';
            }
        }
    </script>
</body>
</html>
    """
    return html_content

# 5. 预测接口
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)
    predictions = model.predict(img_array)
    
    pred_class = class_names[np.argmax(predictions)]
    confidence = f"{float(np.max(predictions)) * 100:.2f}%"
    
    # 强制设置响应编码为UTF-8，防止中文乱码
    return JSONResponse(
        content={"predicted_class": pred_class, "confidence": confidence},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)