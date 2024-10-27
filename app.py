import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyrebase
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from sklearn.exceptions import DataConversionWarning
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, IntegerField
from wtforms.validators import InputRequired, Length, ValidationError, DataRequired
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secretkey'

#
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

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

model_species = load_model('crop (1).h5')
model_species.load_weights('crop_weights (1).h5')

model_segment = load_model('leaf.h5')
model_segment.load_weights('leaf_weights.weights.h5')

f2 = open("drugTree2.pkl", "rb")
model_med = pickle.load(f2)

#Mapping Symptoms as indexes in dataset using dictionary
symptom_mapping = {
    'acidity': 0,
    'indigestion': 1,
    'headache': 2,
    'blurred_and_distorted_vision': 3,
    'excessive_hunger': 4,
    'muscle_weakness': 5,
    'stiff_neck': 6,
    'swelling_joints': 7,
    'movement_stiffness': 8,
    'depression': 9,
    'irritability': 10,
    'visual_disturbances': 11,
    'painful_walking': 12,
    'abdominal_pain': 13,
    'nausea': 14,
    'vomiting': 15,
    'blood_in_mucus': 16,
    'Fatigue': 17,
    'Fever': 18,
    'Dehydration': 19,
    'loss_of_appetite': 20,
    'cramping': 21,
    'blood_in_stool': 22,
    'gnawing': 23,
    'upper_abdomain_pain': 24,
    'fullness_feeling': 25,
    'hiccups': 26,
    'abdominal_bloating': 27,
    'heartburn': 28,
    'belching': 29,
    'burning_ache': 30
}


def process_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def predict_crop_species(img):
    class_labels = ['Aloevera-Aloe barbadensis',
                    'Amla-Phyllanthus emlica Linn',
                    'Amruta_Balli-Tinospora cordifolia',
                    'Arali-Nerium oleander',
                    'Ashoka-Saraca asoca',
                    'Ashwagandha-Withania somnifera',
                    'Avacado-Persea americana',
                    'Bamboo-Bambusoideae',
                    'Basale-Basella alba',
                    'Betel-Piper betle',
                    'Betel_Nut-Areca catechu',
                    'Brahmi-Bacopa monnieri',
                    'Castor-Ricinus communis',
                    'Curry_Leaf-Murraya koenigii',
                    'Doddapatre-Plectanthus amboinicus',
                    'Ekka-Calotropis gigantea',
                    'Ganike-Solanum nigrum',
                    'Gauva-Psidium guajava',
                    'Geranium-Pelargonium',
                    'Henna-Lausonia inermis',
                    'Hibiscus-Hibiscus rosa sinensis',
                    'Honge-Milletia',
                    'Insulin',
                    'Jasmine-Jasmium',
                    'Lemon-Citrus limon',
                    'Lemon_grass-Cymbopogon citratus',
                    'Mango-Mangifera indica',
                    'Mint-Mentha',
                    'Nagadali-Ruta graveolens',
                    'Neem-Azadirachta indica',]
    predictions = model_species.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class


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


def medicineValidation(selectedOptions):
    inputs = np.array(selectedOptions).reshape(1, -1)
    recommend_Med = model_med.predict(inputs)
    return recommend_Med[0]


@app.route('/')
def index():
    return render_template('/login_page.html')


@app.route('/signup')
def signup():
    return render_template('signup_page.html')

@app.route('/login')
def login():
    return render_template('login_page.html')


@app.route('/forgot')
def forgot():
    return render_template('forgot.html')


@app.route('/verification')
def verification():
    return render_template('verification.html')


@app.route('/home')
def home():
    return render_template('home_page.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/plant_species', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = process_image(file_path)
        prediction = predict_crop_species(img)

        plant_species = prediction
        plant_description = fetch_plant_description(plant_species)

        return render_template('prediction.html', prediction=prediction, plant_description=plant_description) 


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
