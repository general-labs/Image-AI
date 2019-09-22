"""
Image AI

- Detect politicial figures (person) using face detection.
- Face priority image crop
- Predict what image it is. Detect object and it's name.
- Predict what image it is using several different image datasets. Detect object and it's name.
- Detect scene or environment info.

"""
from flask import Flask, jsonify, request, url_for, render_template
import json
from flask_cors import CORS

import urllib.request
import imageclassifier
import imageclassifierall

import imagecaption
import face

import subprocess
import random

# JSON display handler.
def display_msg(request_obj, input_obj):
    post_content = request_obj.args.get('url')
    if not request_obj.args.get('url'):
        post_content = request_obj.form['url']
    if not post_content:
        return jsonify({"Error": 'No URL entered'})
    try:
        return jsonify(input_obj(post_content))
    except Exception as e:
        return jsonify({"Error": 'There was an error while processing your request. ' + str(e)})

# Display cropped image in page.
def inline_crop_display(post_content):
    random_num = random.randint(1000, 9999)
    if not post_content:
        return 'No URL entered'
    try:
        with urllib.request.urlopen(post_content) as url:
            with open('static/temp.jpg', 'wb') as f:
                f.write(url.read())
       
        cmd_output = subprocess.call(['python3', 'intelligentcrop.py', str(random_num), 'ayan'])
        data_val = {'random_num':str(random_num) }
        return '''
    <html>
        <head>
            <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
            <meta http-equiv="Pragma" content="no-cache" />
            <meta http-equiv="Expires" content="0" />
            <title>Home Page - Smart Crop</title>
        </head>
        <body>
            <h1>Smart Crop by Content AI</h1>
            <img src = "static/output''' + data_val['random_num'] + '''.jpg">
        </body>
    </html>'''

    except:
        return 'There was an error while processing your request.'

# JSON cropped image URL handler.
def display_cropped_url(post_content):
    random_num = random.randint(1000, 9999)
    with urllib.request.urlopen(post_content) as url:
        with open('static/temp.jpg', 'wb') as f:
            f.write(url.read())
    cmd_output = subprocess.call(['python3', 'intelligentcrop.py', str(random_num), 'ayan'])
    return {"url": 'static/output' + str(random_num) + '.jpg'}

# Web Server declaration.
def flask_app():
    app = Flask(__name__)
    CORS(app)

    # Root route.
    @app.route('/', methods=['GET', 'POST']) #allow both GET and POST requests
    def form_example():
        return '''
                 <h1>Content Classifier</h1>
                 <form method="POST" action = "/predict_image">
                      <p> Copy paste data from web </p>
                      <textarea name = "content"  rows="20" cols="50"></textarea>
                      <input type="submit" value="Submit"><br>
                  </form>'''

    # Smart crop display route.
    @app.route('/smart_crop_display')
    def smart_crop():
        return inline_crop_display(request.args.get('url'))
        
    # Image prediction route.
    @app.route('/predict_image', methods=['GET', 'POST'])
    def start():
        return display_msg(request, imageclassifier.classify_image)

    # Display all four model route.
    @app.route('/predict_image_all', methods=['GET', 'POST'])
    def predict_image_all():
        return display_msg(request, imageclassifierall.classify_image)

    # Image caption route.
    @app.route('/image_caption', methods=['GET', 'POST'])
    def image_caption():
        return display_msg(request, imagecaption.gen_caption)

    # Detect person in an image route.
    @app.route('/detect_person', methods=['GET', 'POST'])
    def face_detection():
        return display_msg(request, face.find_person)

    # Smart crop route.
    @app.route('/smart_crop', methods=['GET', 'POST'])
    def smartcrop():
        return display_msg(request, display_cropped_url)

    return app

# Initiate Web Server
if __name__ == '__main__':
    app = flask_app()
    app.run(debug=False, host='0.0.0.0', port=5002)
