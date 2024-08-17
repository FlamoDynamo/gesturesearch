import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Tải mô hình
model = load_model("models/sign_language_model.h5")

def predict(data):
    data = np.array(data).reshape(1, -1)  # Đảm bảo dữ liệu đúng định dạng
    prediction = model.predict(data)
    return prediction[0].tolist()  # Chuyển đổi về dạng list nếu cần