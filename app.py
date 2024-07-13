from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import base64
import cv2
app = Flask(__name__)
model = load_model('my_model.h5')  # Đường dẫn tới mô hình đã lưu


def preprocess_image(image, target_size):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    # Flatten the image to match the model's expected input shape
    image = image.flatten().reshape(1, -1)  # Correctly reshape the image here
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        if 'image' in data:
            image_data = data['image']
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            processed_image = preprocess_image(image, target_size=(28, 28))
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction, axis=1)[0]
            return jsonify({'label': int(predicted_label)})
        return jsonify({'error': 'No image data'}), 400
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
