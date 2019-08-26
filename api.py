from flask import Flask, request, jsonify, session
import pandas as pd
import os, json, time
from werkzeug.utils import secure_filename
from flask_cors import CORS
from utils import prediction,train
from rq import Queue
from rq.job import Job
from worker import conn

app = Flask(__name__)
CORS(app)
CORS(app,supports_credentials=True)
app.secret_key = "sjfdxcvbfd,mxhfm vhsdlxj,fhsj"
q = Queue('low',connection=conn)
sc_array = []

@app.route('/upload', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        fName = f.filename
        f.save(os.path.join(app.instance_path,secure_filename(fName)))
        dataset = pd.read_excel(f)
        session['file-name'] = fName
        session['input-para'] = dataset.values[0].size - 5 - 1
        session['output-para'] = 5 #maximum value
        # print(vars(session))
        return "File Uploaded Successfully"
    return "Method is not supported"

@app.route('/neurons', methods=['POST'])
def getNeurons():
    fileName = ""
    if request.method == 'POST':

        if 'input-para' in session:
            input_para = session['input-para']
        
        print(session['file-name'])

        return jsonify(input_para)
    return "Method is not supported"

@app.route('/controlData', methods=['POST'])
def setControlData():
    fileName = ""
    if request.method == 'POST':
        if 'file-name' in session:
            fileName = session['file-name']
        if 'input-para' in session:
            input_para = session['input-para']
        if 'output-para' in session:
            output_para = session['output-para']
        
        print(session['file-name'])
        ta = request.form["ta"]
        tf = request.form["tf"]
        neurons = request.form["neurons"]

        session["ta"] = ta
        session["tf"] = tf
        session["neurons"] = neurons

        f = os.path.abspath("instance/"+fileName)
        
        job = q.enqueue(train,output_para,input_para,f,neurons,tf,ta)
        return job.get_id()

    return "Method is not supported"

@app.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):

    job = Job.fetch(job_key, connection=conn)

    if job.is_finished: 
        return jsonify(job.result)
    else:
        return "202"

@app.route('/getParam', methods=['POST'])
def getParam():
    fileName = ""
    if request.method == 'POST':
        if 'file-name' in session:
            fileName = session['file-name']
        if 'input-para' in session:
            input_para = session['input-para']
        if 'output-para' in session:
            output_para = session['output-para']
        
        print(session['file-name'])

        f = os.path.abspath("instance/"+fileName)
        
        data_xls = pd.read_excel(f)
        cols_input = data_xls.columns[1 + 5: 1 + 5 + input_para]
        cols_output = data_xls.columns[1:output_para+1]
        arr1 = []
        arr2 = []

        for col in cols_input:
            arr1.append(col)
        for col in cols_output:
            if "Unnamed" in col:
                continue
            arr2.append(col)
        
        arr = []
        arr.append(arr1)
        arr.append(arr2)

        return jsonify(arr)
        
    return "Method is not supported"

@app.route('/predict', methods=['POST'])
def predict():
    fileName = ""
    if request.method == 'POST':
        if 'file-name' in session:
            fileName = session['file-name']
        if 'input-para' in session:
            input_para = session['input-para']
        if 'output-para' in session:
            output_para = session['output-para']
        
        print(session['file-name'])
        
        f = os.path.abspath("instance/"+fileName)
        data_xls = pd.read_excel(f)
        cols_input = data_xls.columns[1 + output_para: 1 + output_para + input_para]

        arr = []
        for col in cols_input:
            arr.append(float(request.form[col]))
        
        print(arr)

        job = q.enqueue(prediction,arr,output_para,f,output_para)
        return job.get_id()          

    return "Method is not supported"



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
