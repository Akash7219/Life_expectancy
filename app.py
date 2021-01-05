from flask import Flask,request,render_template
import joblib
import numpy as np
import sklearn
import pickle

filename="gradient.pkl"
model=joblib.load(open(filename,'rb'))

app=Flask(__name__)


@app.route("/")
def home():
    return render_template("result.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        adult_mortality=float(request.form["adult_mortality"])
        alcohol=float(request.form["alcohol"])
        percentage_expenditure=float(request.form["percentage_expenditure"])
        hepatitisB=float(request.form["hepatitisB"])
        measles=int(float(request.form["measles"]))
        bmi=float(request.form["bmi"])
        under_five_deaths=float(request.form["under_five_deaths"])
        polio=float(request.form["polio"])
        total_expenditure=float(request.form["total_expenditure"])
        hiv_aids=float(request.form["hiv_aids"])
        gdp=float(request.form["gdp"])
        thinness_10_19=float(request.form["thinness_10-19"])
        income_composition_of_resources=float(request.form["income_composition_of_resources"])
        status=(request.form["status"])
        if status=="Developing":
            status_num_0=1
            status_num_1=0
        else:
            status_num_0=0
            status_num_1=1
            
       
        prediction_list=[adult_mortality,
                         alcohol,
                         percentage_expenditure,
                         hepatitisB,
                         measles,
                         bmi,
                         under_five_deaths,
                         polio,
                         total_expenditure,
                         hiv_aids,
                         gdp,
                         thinness_10_19,
                         income_composition_of_resources,
                         status_num_0,
                         status_num_1]
        predict_array=np.array([prediction_list])
        prediction_model=model.predict(predict_array)
        
        output=round(prediction_model[0],2)
        if output<0:
            return render_template('result.html',prediction_texts="Sorry you cannot predict the output")
        else:
            return render_template('result.html',prediction_text="The life expectancy of this country is {}".format(output))
    else:
        return render_template('result.html')
if __name__=="__main__":
    app.run(debug=True,use_reloader=False)

        

        
    
