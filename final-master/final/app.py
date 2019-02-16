# Example adapted from http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
# @NOTE: The code below is for educational purposes only.
# Consider using the more secure version at the link above for production apps
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
import os
from keras.models import load_model
from flask import Flask, request, render_template,json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


def train_model():
    arr = []
    df = pd.read_csv("stats1.csv")
    target_names = df["TD"]
    data = df.drop("TD", axis=1)
    feature_names = data.columns

    data = df.values
    X = data[:, 0:11]
    y = data[:, 11]
    y = y.astype('int')
    X = X.astype('int')

    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    # print(f"R2 Score: {score}")
    arr.append(X)
    arr.append(y)
    return arr


@app.route('/data')
def landing_page():

    df = pd.read_csv("stats2.csv")


    # a=[]
    # a.append(df.to_json(orient='records', lines=True))
    # a
    response = app.response_class(
    # response=json.dumps(df.to_json(orient='index')),
    response=df.to_json(orient='index'),

    status=200,
    mimetype='application/json'
    )
    return response




@app.route('/', methods=['GET', 'POST'])
def webprint():
    arr = train_model()

    # load model
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(arr[0], arr[1])
    print(result)
    # make prediction
    li = []

    atemp = request.form.get('att')
    comp = request.form.get('cmp')
    perc = request.form.get('pct')
    yarsd = request.form.get('yds')
    yardsatt = request.form.get('ypa')
    inter = request.form.get('inter')
    interper = request.form.get('intpct')
    longth = request.form.get('lg')
    sack = request.form.get('sack')
    loss = request.form.get('loss')
    rate = request.form.get('rate')
    tchdper = request.form.get('tprc')

    li.append(atemp)
    li.append(comp)
    li.append(perc)
    li.append(yarsd)
    li.append(yardsatt)
    li.append(tchdper)
    li.append(inter)
    li.append(interper)
    li.append(longth)
    li.append(sack)
    li.append(loss)
    li.append(rate)
    mat = np.array(li)

    x = np.array(['1.1', '2.2', '3.3'])
    y = x.astype(np.float)

    mat_con = mat.astype(np.float)

    # print(loaded_model.predict([mat_con]))


    if request.method =='POST':
        # print(loaded_model.predict([y]))  
        print(mat_con)
        model = load_model("td_predict.h5")


        val = model.predict([[mat_con]])
        data = {"Predicted Touchdowns":str(val), "Model Type": "Sequential","Loaded Model":"td_predict.h5", "Epochs": "5000"}
        print(data)

        response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
        )
        return response


        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)