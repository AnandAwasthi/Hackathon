from flask import Flask, request, redirect, url_for, flash, jsonify, send_from_directory, render_template
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import scipy
import json
from matplotlib import cm as CM
from model import CCNN
import torch
from torchvision import transforms
import numpy as np
import math

app = Flask(__name__)

def predict_crowd_count(model, img_path: str):
    img = transform(Image.open(img_path).convert('RGB'))
    output = model(img.unsqueeze(0))
    return round(abs(output.detach().cpu().sum().numpy()))


@app.route('/')
def upload_form():
	return render_template('CrowdScannerV1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    offset = 10 
    f = request.files.get('file')
    f.save(UPLOAD_DIRECTORY + f.filename)
    crowd_count1 = predict_crowd_count(model1, UPLOAD_DIRECTORY + f.filename)
    crowd_count2 = predict_crowd_count(model2, UPLOAD_DIRECTORY + f.filename)
    mean_crowd = math.ceil(np.mean([crowd_count1, crowd_count2]))
    crowd_count_red =  mean_crowd - offset
    if crowd_count_red > 0 :
        mean_crowd = crowd_count_red
    print(mean_crowd)
    return str(mean_crowd)

    
if __name__ == '__main__':
        
    transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
    device = torch.device('cpu')              
    model1 = CCNN()
    model2 = CCNN()
    checkpoint1 = torch.load('./a_ccih.pth.tar', map_location=device)
    model1.load_state_dict(checkpoint1['state_dict'])
    checkpoint2 = torch.load('./aa_ccih.pth.tar', map_location=device)
    model2.load_state_dict(checkpoint2['state_dict'])
    UPLOAD_DIRECTORY = './'
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)
    app.run(debug = True, host= 'localhost', port = 5000)

