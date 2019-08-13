from flask import Flask, request, jsonify
import pandas as pd
import os, json
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/upload', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        inp_par = 6
        f = request.files['file']
        fName = f.filename
        f.save(os.path.join(app.instance_path,secure_filename(fName)))
        data_xls = pd.read_excel(f)
        print(data_xls)
        cols = data_xls.columns[1:inp_par]
        arr = []
        for col in cols:
            arr.append(col)
        return jsonify(arr)
    return "Method is not supported"

if __name__ == '__main__':
    app.run(debug=True)