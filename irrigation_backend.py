from firebase import firebase
from flask import Flask,jsonify,request
from flask_ngrok import run_with_ngrok
import pickle
import random
import cv2
import numpy as np

import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite

from PIL import Image
import requests

app = Flask(__name__)
run_with_ngrok(app)
firebase = firebase.FirebaseApplication('credentials_to_firebase', None)
@app.route('/1')
def index():
    return "Hello"

@app.route('/crop', methods=['GET'])
def upload_file():
  result = firebase.get('/users1','')
  lst=[]
  for key in result.keys():
    lst.append(result[key])
  print(lst)
  filename = '/content/NBClassifier.pkl'
  loaded_model = pickle.load(open(filename, 'rb'))
  result = loaded_model.predict([lst])
  print(result)
  lst=[result[0]]
  test_list = ["Onion","Pumpkin","Grapes","Manioc","Tobacoo"]
  rand_idx = random.randrange(len(test_list))
  random_num = test_list[rand_idx]
  lst.append(random_num)
  print(lst)
  return jsonify(lst)

@app.route('/fertilizer', methods=['GET'])
def fertilizer():
  result = firebase.get('/users1','')
  lst=[]
  for key in result.keys():
    lst.append(result[key])
  print(lst)
  filename = '/content/NBClassifier (1).pkl'
  loaded_model = pickle.load(open(filename, 'rb'))
  result = loaded_model.predict([lst])
  print(result)
  return jsonify(str(result[0]))


@app.route('/water', methods=['GET'])
def water_needed():
  result = firebase.get('/users1','')
  lst=[]
  for key in result.keys():
    lst.append(result[key])
  print(lst)
  lst.append(8)
  filename = '/content/RFClassifier.pkl'
  loaded_model = pickle.load(open(filename, 'rb'))
  result = loaded_model.predict([[3.23,78.5,6.78,8]])
  print(result)
  return jsonify(str(result[0]))

@app.route('/submit', methods=['POST','GET'])
def image_upload():
    dict1={}
    data = request.json
    uploaded_file = data['image']
    print(uploaded_file)
    img_data = requests.get(uploaded_file).content
    with open('image_name.jpg', 'wb') as handler:
      handler.write(img_data)
    # uploaded_file = request.files['image']
    # if uploaded_file.filename != '':
    #     uploaded_file.save(uploaded_file.filename)
    # path="/content/"+str(uploaded_file.filename)
    path='/content/image_name.jpg'
    print(path)
    # q= request.files['image']
    def load_labels(label_path):
        r"""Returns a list of labels"""
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]


    def load_model(model_path):
        r"""Load TFLite model, returns a Interpreter instance."""
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter


    def process_image(interpreter, image, input_index, k=3):
        r"""Process an image, Return top K result in a list of 2-Tuple(confidence_score, label)"""
        input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

        # Process
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # Get outputs
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data.shape)  # (1, 1001)
        output_data = np.squeeze(output_data)

        # Get top K result
        top_k = output_data.argsort()[-k:][::-1]  # Top_k index
        result = []
        for i in top_k:
            score = float(output_data[i] / 255.0)
            result.append((i, score))

        return result


    def display_result(top_result, frame, labels):
        r"""Display top K result in top right corner"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.6
        color = (255, 0, 0)  # Blue color
        thickness = 1

        for idx, (i, score) in enumerate(top_result):
            # print('{} - {:0.4f}'.format(label, score))
            x = 12
            y = 24 * idx + 24
            dict1[labels[i]]=score
            cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                        (x, y), font, size, color, thickness)
            

        # cv2_imshow(frame)

    model_path = '/content/model.tflite'
    label_path = '/content/labels.txt'
    image_path = path

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(frame.shape)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height))

    top_result = process_image(interpreter, image, input_index)
    display_result(top_result, frame, labels)
    # return jsonify(dict1)
    for k,v in dict1.items():
      response=k
    return jsonify(k[2:])
app.run()