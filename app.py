from flask import Flask, render_template, request
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np

#load model
densenet_model = tf.keras.models.load_model('densenet_cub.h5')
yolo_model = YOLO('best.pt')

app = Flask(__name__)

def load_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    yolo_result = None
    img_path = None

    if request.method == 'POST':
        img_file = request.files['image']
        img_path = "static/" + img_file.filename
        img_file.save(img_path)

        # DenseNet
        img_array = load_image(img_path)
        predictions = densenet_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_labels = ['Healthy', 'Unhealthy']

        result = class_labels[predicted_class]

        if result == "Unhealthy":
            yolo_result = yolo_model(img_path)
            # Lưu kết quả của YOLO
            yolo_result[0].save('static/yolo_output.jpg')

    return render_template('index.html', image=img_path, result=result, yolo_result=yolo_result)

if __name__ == '__main__':
    app.run(debug=True)
