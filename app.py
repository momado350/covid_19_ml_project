# import all dependencies
from flask import Flask, jsonify, render_template,request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import pickle


#======================================================
# Flask app
#======================================================
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
ENV = 'prod'

if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/zebra_db'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = ''
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# set up a database instance

# create our web route
@app.route("/",methods=['POST', 'GET'])
def main():
    # handle user inputs
#https://stackoverflow.com/questions/56934303/assign-a-variable-from-html-input-to-python-flask  
    try:
        state = request.form.get('state')
        zipcode = request.form.get('zipcode')
        wcases = int(request.form.get('weeklycases'))
        wcaserate = int(request.form.get('weeklycaserate'))
        wtests = int(request.form.get('weeklytests'))
        wdeathrate = float(request.form.get('weeklydeathrate'))
    except:
        state = "Null"
        zipcode = "0"
        wcases = 0
        wcaserate = 0
        wtests = 0
        wdeathrate = 0.0
    global output 
    output=[wcases,wcaserate,wtests,wdeathrate]
    
    # scale data
    from sklearn.preprocessing import StandardScaler

    X_test =output

    X_test =np.asarray(X_test)
    X_test =X_test.reshape(1,-1)

    # make prediction using user input data
    predictions = model.predict(X_test)

    # render html with features enterd and prediction made
    return render_template("index.html",output=output, predictions=predictions,state=state,zipcode=zipcode)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# run flask app
# app is under development that is why we still using dev env with debug=True
if __name__ == "__main__":
    app.run(debug=True)