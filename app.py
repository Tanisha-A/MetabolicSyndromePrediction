import numpy as np
from flask_cors import CORS
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template,json
from sklearn import model_selection


from xgboost import XGBClassifier

app = Flask(__name__, template_folder = '.')
CORS(app)

@app.route('/')
def home():
    return render_template('C:\\Users\\user\\Downloads\\MP\\getstarted.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = json.loads(request.data)
    age = data.get('age')
    sex = data.get('sex')
    marital = data.get('marital')
    income = data.get('income')
    race = data.get('race')
    waistcirc = data.get('waistcirc')
    bmi = data.get('bmi')
    albuminuria = data.get('albuminuria')
    uralbcr =data.get('uralbcr')
    uricacid = data.get('uricacid')
    bloodglucose =data.get('bloodglucose')
    hdl = data.get('hdl')
    triglycerides = data.get('triglycerides')
    info= [[age, sex, marital, income, race, waistcirc, bmi,albuminuria,uralbcr, uricacid, bloodglucose, hdl,triglycerides]]
    
    # Create the pandas DataFrame
    test = pd.DataFrame(info, columns=['age', 'sex', 'marital', 'income', 'race', 'waistcirc', 'bmi', 'albuminuria', 'uralbcr', 'uricacid', 'bloodglucose', 'hdl','triglycerides'])
    
    # Load PreProcessor
    pp = pickle.load(open("C:\\Users\\user\\Downloads\\MP\\pp.p","rb"))
    
    # Apply Preprocessing to Input
    test_processed = pp.transform(test)
    testdf = pd.DataFrame(test_processed)

    #Load Model
    loaded_model = XGBClassifier(max_depth = 2)
    loaded_model.load_model("C:\\Users\\user\\Downloads\\MP\\prediction.json")
    
    #Make Prediction
    y_pred = loaded_model.predict(testdf)


    if y_pred[0]==1:
        print("Negative")
        return jsonify("Prediction: Patient has a low risk of Metabolic Syndrome.")
    else:
        print("Positive") 
        return jsonify("Prediction: Patient has a high risk of Metabolic Syndrome.")


if __name__ == "__main__":
    app.run(debug = True)