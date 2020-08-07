from flask import Flask, jsonify, render_template,request


import numpy as np
import pandas as pd
import pickle


#======================================================
# Flask app
#======================================================
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/",methods=['POST', 'GET'])
def main():
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
    

    from sklearn.preprocessing import StandardScaler

    X_test =output

    X_test =np.asarray(X_test)
    X_test =X_test.reshape(1,-1)

    predictions = model.predict(X_test)

    return render_template("index.html",output=output, predictions=predictions,state=state,zipcode=zipcode)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    app.run(debug=True)