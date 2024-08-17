from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Tải mô hình từ file .h5
model = load_model('models/sign_language_model.h5')

# Tải trọng số từ file .pkl
weights = joblib.load('models/sign_language_weights.pkl')

# Cập nhật trọng số cho mô hình
model.set_weights(weights)

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        data = request.json

        # Kiểm tra định dạng dữ liệu
        if data is None or 'features' not in data:
            return jsonify({"error": "Invalid input"}), 400

        features = data['features']

        # Chuyển đổi thành numpy array và reshape về kích thước (1, 28, 28, 1)
        if len(features) != 784:  # Kiểm tra độ dài đầu vào
            return jsonify({"error": "Features must have length of 784"}), 400
        
        features = np.array(features).reshape(1, 28, 28, 1)

        # Dự đoán
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=-1).tolist()  # Nhận lớp dự đoán

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)