from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


app = Flask(__name__)
CORS(app)


model = None


CLASS_NAMES = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']


def prepare_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128, 3))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    return input_arr

@app.route('/')
def home():
    return 'Flask server is running and TensorFlow is ready!'

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = tf.keras.models.load_model('model/my_model.keras')
    
    print('Inside the prediction endpoint')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    

    image_file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    filename = secure_filename(image_file.filename)
    filepath = os.path.join("uploads", filename)
    print(filepath)
    image_file.save(filepath)
    try:
        image = prepare_image(filepath)
        prediction = model.predict(image)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return jsonify({
            'predicted_class_index': predicted_class_index,
            'predicted_class_name': predicted_class_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath) 


#  soyabean disease 
interpreter = tf.lite.Interpreter(model_path="model/soya_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class_names = ['rust disease','healthy']

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256, 3))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    return input_arr

@app.route('/predict/soya', methods=['POST'])
def predict_soya():
    print('Inside the Soya prediction endpoint')

    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    filename = secure_filename(image_file.filename)
    filepath = os.path.join("uploads", filename)
    image_file.save(filepath)

    try:
        image = preprocess_image(filepath)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = int(np.argmax(prediction))
        predicted_class_name = class_names[predicted_class_index]

        return jsonify({
            'predicted_class_index': predicted_class_index,
            'predicted_class_name': predicted_class_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        os.remove(filepath)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
