import torch


from flask import Flask, render_template, request
from model import model, preprocess, predict, plot_pred_images

from flask import Flask, request, render_template
import os
import cv2
import base64
import numpy as np

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import base64
import os


UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/', methods=['POST'])
# def upload_file():
#     file = request.files['file']
#     filename = file.filename
#     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
#     return render_template('result.html', file_path=file_path)


@app.route("/")
def main():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def predict_image_file():

    try:
    
        if request.method == 'POST':
            file = request.files['file']
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            seg_mask = predict(file_path, model)
            # img = (request.files['file'].stream)
            # output = predict(img, model)
            pred = plot_pred_images(preprocess(file_path), seg_mask)
            return render_template("result.html", predictions=pred)

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)

if __name__ == "__main__":
    app.run(port=80, debug=True)
