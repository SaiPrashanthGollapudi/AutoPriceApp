import flask
from flask import request, render_template
import joblib

app = flask.Flask(__name__,static_url_path="")

@app.route('/', methods=['GET'])
def sendIndex():
        return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
        wb=float(request.form['wb'])
        lent=float(request.form['lent'])
        wid=float(request.form['wid'])
        fs=float(request.form['fs'])
        bore=float(request.form['bore'])
        hp=float(request.form['hp'])
        cmpg=float(request.form['cmpg'])
        X = [[wb,lent,wid,fs,bore,hp,cmpg]]
        model = joblib.load('deployment_auto.pkl')
        species = model.predict(X)[0]
        return render_template('predict.html',predict=species)
app.run()
