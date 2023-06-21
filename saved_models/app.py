from flask import Flask, request, render_template
from flask import Response
from datetime import datetime
from flask import jsonify
import pandas
import pickle
import json
import requests
import os
import csv
from tabulate import tabulate

from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np




headers = {"Authorization": "Bearer ya29.a0ARrdaM_YkI6oUm949UJteFylUpoLGG114jpBLlEiTSJZkfPSqwPaUcWmJKHRN9aPNBpOZoXdbCjC5BRezFaooZSVvfFqMyKkbmb_ZuuxrzkARnNJh06-Dm-xvq4FVlpmnoBm1IF2n4seBnhRjV79Si4XmNS7"}

model_path = 'saved_models/model.pkl'
model = pickle.load(open(model_path, 'rb'))


model_pathLG = 'saved_models/modelLG.pkl'
modelLG = pickle.load(open(model_pathLG, 'rb'))

model_pathLSTM = 'saved_models/modelXG.pkl'
modelLSTM = pickle.load(open(model_pathLG, 'rb'))

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/modelcatboost", methods=['POST','GET'])
def modelcat():
        #  data=pandas.read_csv(r'D:\Amaluttan\Money Laudering\saved_models\predictions\rr.csv')
        # #  fraud_data=data[10:20] 
        #  fraud_data = data[data['isFraud'] == "Fraud"]
         return render_template('modelcatboost.html')

@app.route("/modelkeras", methods=['POST','GET'])
def modelkeras():
         return render_template('modelkeras.html')


@app.route("/modellight", methods=['POST','GET'])
def modellight():
         return render_template('modellightgbm.html')

@app.route("/CatBoostClassifier", methods=['POST','GET'])
def catboostclass():
        #  if request.method == "post":
        #        transfertype=request.form['inlineRadioOptions']
        #        amount=request.form['amount']
        #        orgoldbal=request.form['orgoldbal']
        #        orgnewbal=request.form['orgnewbal']
        #        desoldbal=request.form['desoldbal']
        #        desnewbal=request.form['desnewbal']
        #        acctype=request.form['Acctype']
        #        features=np.array([3,22,transfertype,amount,orgoldbal,orgnewbal,desoldbal,desnewbal,acctype])
        #        res="ops"
        #        if model.predict(features)==0:
        #           print('No Fraud')
        #           res="No Fraud"
        #        else:
        #           print('Fraud')
        #           res="Fraud"
        #        print(model.predict(features))
        #        return render_template('CatboostIndividual.html', result=res)
         return render_template('CatboostIndividual.html')




@app.route("/predict", methods=['POST','GET'])
def predictRouteClient():

        if request.form is not None:
            path = request.form['filepath']
            data = pandas.read_csv(path,error_bad_lines=False)
            y_pred = model.predict(data)
            # print(y_pred)
            data['isFraud'] = y_pred
            # print(data)
            # print(tabulate(data.head(), headers='keys', tablefmt='psql'))
            #path = os.getcwd()
            #output_path = os.path.join(path,'rr.csv')
            # correlation = data.corr()
            # print(correlation['isFraud'].sort_values(ascending=False))
            data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
            data.to_csv(r'D:\Amaluttan\Money Laudering\saved_models\predictions\rr.csv', index=False)
            data.set_index("isFraud",inplace = True)
            bro=[]
            with open('data.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    bro.append(row)
            # features = np.array([0,164,5,177896.19,188357,0,2506,3262737.42,1])
            features=np.array([3,22,1,2068118.36,73758,2068118.36,3460,3054285.45,0])

            if model.predict(features)==0:
                  print('No Fraud')
            else:
                  print('Fraud')
            print(model.predict(features))
 
# Using the operator .loc[]
# to select multiple rows
            # result = data.loc[["Fraud"]]
            # print(result)
            result2 = data.loc[["No Fraud"]]
            print(result2)
 
            file_name = "Output.csv" 

    
            return file_name
        else:
            print('Nothing Matched')
    











@app.route("/get_data", methods=['POST','GET'])
def get_data():
    
            # path = request.form['filepath']
            path=request.form.get('filepath')
            type=request.form.get('type')
            print(path)
            print(type)
            data = pandas.read_csv(path,error_bad_lines=False)
            y_pred = model.predict(data)
            data['isFraud'] = y_pred
            data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
            data.to_csv(r'D:\Amaluttan\Money Laudering\saved_models\predictions\rr.csv', index=False)
            data.set_index("isFraud",inplace = True)
            bro=[]
            with open(r'D:\Amaluttan\Money Laudering\saved_models\predictions\rr.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['isFraud'] == type:
                        bro.append(row)
            r="opps"
            return jsonify(bro)
    

            print('Nothing Matched')











    # data = pandas.read_csv(r'D:\Amaluttan\Money Laudering\saved_models\predictions\rr.csv')  # read data from CSV file
    # data_json = data.to_json(orient="records")  # convert data to JSON
    # return jsonify(data_json)  # return JSON object


@app.route("/keras", methods=['POST','GET'])
def get_keras():
      
      if request.form is not None:
            

        type = request.form['type']
        amount = request.form['amount']
        oldbalance = request.form['oldbalance']
        newbalance = request.form['newbalance']
        features = np.array([[type,amount,oldbalance, newbalance]])
        res=""
        res=modelLG.predict(features)
        print(res[0])  
        

        # response_data = {'success': True}
      return res[0]

@app.route("/process", methods=['POST','GET'])
def process():
    transfertype=request.form.get('inlineRadioOptions')
    amount=request.form.get('amount')
    orgoldbal=request.form.get('orgoldbal')
    orgnewbal=request.form.get('orgnewbal')
    desoldbal=request.form.get('desoldbal')
    desnewbal=request.form.get('desnewbal')
    acctype=request.form.get('Acctype')
    # output = {
    #     'name': name,
    #     'age': age,
    #     'email': email
    # }
    features=np.array([0,22,transfertype,amount,orgoldbal,orgnewbal,desoldbal,desnewbal,acctype])
    print(features)
    res="ops"
    if model.predict(features)==0:
        print('No Fraud')
        res="Not Fraudulent"
    else:
        print('Fraud')
        res="Fraudulent"
    print(model.predict(features))
    output_data = res.upper()

    compare_data = {}
    with open('reports/catboostperfo.csv', 'r') as file:
        reader = csv.reader(file)
        keys = next(reader)
        values = next(reader)
        compare_data = dict(zip(keys, values))
    print(compare_data)
    # return jsonify(data)


    return jsonify({'output_data': output_data,'compare_data':compare_data})  # return JSON object




@app.route("/lstmmodel", methods=['POST','GET'])
def lstmprocess():
    transfertype=request.form.get('inlineRadioOptions')
    amount=request.form.get('amount')
    orgoldbal=request.form.get('orgoldbal')
    orgnewbal=request.form.get('orgnewbal')
    desoldbal=request.form.get('desoldbal')
    desnewbal=request.form.get('desnewbal')
    acctype=request.form.get('Acctype')
    # output = {
    #     'name': name,
    #     'age': age,
    #     'email': email
    # }
    features=np.array([0,22,transfertype,amount,orgoldbal,orgnewbal,desoldbal,desnewbal,acctype]).reshape(1, -1)
    features = features.astype(float) 
    print(features)
    res="ops"
    predict=modelLSTM.predict(features)
    if predict[0]==0:
        print('No Fraud')
        res="Not Fraudulent"
    else:
        print('Fraud')
        res="Fraudulent"
    print(modelLSTM.predict(features))
    output_data = res.upper()

    compare_data = {}
    with open('reports/lstmperfo.csv', 'r') as file:
        reader = csv.reader(file)
        keys = next(reader)
        values = next(reader)
        compare_data = dict(zip(keys, values))
    print(compare_data)
    # return jsonify(data)


    return jsonify({'output_data': output_data,'compare_data':compare_data})  # return JSON object



@app.route("/LGmmodel", methods=['POST','GET'])
def LGprocess():
    transfertype=request.form.get('inlineRadioOptions')
    amount=request.form.get('amount')
    orgoldbal=request.form.get('orgoldbal')
    orgnewbal=request.form.get('orgnewbal')
    desoldbal=request.form.get('desoldbal')
    desnewbal=request.form.get('desnewbal')
    acctype=request.form.get('Acctype')
    # output = {
    #     'name': name,
    #     'age': age,
    #     'email': email
    # }
    features=np.array([0,22,transfertype,amount,orgoldbal,orgnewbal,desoldbal,desnewbal,acctype]).reshape(1, -1)
    features = features.astype(float) 
    print(features)
    res="ops"
    predict=modelLG.predict(features)
    if predict[0]==0:
        print('No Fraud')
        res="Not Fraudulent"
    else:
        print('Fraud')
        res="Fraudulent"
    print(modelLG.predict(features))
    output_data = res.upper()

    compare_data = {}
    with open('reports/lightperfo.csv', 'r') as file:
        reader = csv.reader(file)
        keys = next(reader)
        values = next(reader)
        compare_data = dict(zip(keys, values))
    print(compare_data)
    # return jsonify(data)


    return jsonify({'output_data': output_data,'compare_data':compare_data})  # return JSON object

if __name__ == "__main__":
    app.run()