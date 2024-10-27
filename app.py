import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyrebase
import pickle
from sklearn.exceptions import DataConversionWarning
from flask_wtf import FlaskForm


app = Flask(__name__)


config = {
  "apiKey": "AIzaSyDTU0gsNDqmP6JCzCiXh7mogL1FFeMOLVU",
  'authDomain': "plant-description-7106e.firebaseapp.com",
  'databaseURL': "https://plant-description-7106e-default-rtdb.firebaseio.com",
  'projectId': "plant-description-7106e",
  'storageBucket': "plant-description-7106e.appspot.com",
  'messagingSenderId': "708431906520",
  'appId': "1:708431906520:web:b5e269a8aa4c7162b1f9ad",
  'measurementId': "G-FM1X0MM2DB"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

def fetch_plant_description(plant_species):
    try:

        data = db.child('Plant').child(plant_species).get().val()
        if data:
            return data
        else:

            return {
                'message': 'Data not found for plant species: ' + plant_species
            }
    except Exception as e:

        return {
            'error': str(e)
        }


def fetch_leaf_description(leaf_species):
    try:

        data = db.child('Plant').child(leaf_species).get().val()
        if data:
            return data
        else:

            return {
                'message': 'Data not found for plant species: ' + leaf_species
            }
    except Exception as e:

        return {
            'error': str(e)
        }


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_segment = load_model('leaf.h5')
model_segment.load_weights('leaf_weights.weights.h5')



def process_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img




def predict_leaf_species(img):
    class_labels = ['Alpinia Galanga (Rasna)',
                    'Amaranthus Viridis (Arive-Dantu)',
                    'Artocarpus Heterophyllus (Jackfruit)',
                    'Azadirachta Indica (Neem)',
                    'Basella Alba (Basale)',
                    'Brassica Juncea (Indian Mustard)',
                    'Carissa Carandas (Karanda)',
                    'Citrus Limon (Lemon)',
                    'Ficus Auriculata (Roxburgh fig)',
                    'Ficus Religiosa (Peepal Tree)',
                    'Hibiscus Rosa-sinensis',
                    'Jasminum (Jasmine)',
                    'Mangifera Indica (Mango)',
                    'Mentha (Mint)',
                    'Moringa Oleifera (Drumstick)',
                    'Muntingia Calabura (Jamaica Cherry-Gasagase)',
                    'Murraya Koenigii (Curry)',
                    'Nerium Oleander (Oleander)',
                    'Nyctanthes Arbor-tristis (Parijata)',
                    'Ocimum Tenuiflorum (Tulsi)',
                    'Piper Betle (Betel)',
                    'Plectranthus Amboinicus (Mexican Mint)',
                    'Pongamia Pinnata (Indian Beech)',
                    'Psidium Guajava (Guava)',
                    'Punica Granatum (Pomegranate)',
                    'Santalum Album (Sandalwood)',
                    'Syzygium Cumini (Jamun)',
                    'Syzygium Jambos (Rose Apple)',
                    'Tabernaemontana Divaricata (Crape Jasmine)',
                    'Trigonella Foenum-graecum (Fenugreek)']
    predictions = model_segment.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class



@app.route('/')
def index():
    return render_template('/upload.html')


# @app.route('/upload')
# def upload():
#     return render_template('upload.html')


@app.route('/plant_segment', methods=['POST'])
def plant_segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    file = request.files['image']
    if file.filename == "":
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = process_image(file_path)
        prediction = predict_leaf_species(img)

        leaf_species = prediction
        plant_description = fetch_leaf_description(leaf_species)

        return render_template('prediction.html', prediction=prediction, plant_description=plant_description) #


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
