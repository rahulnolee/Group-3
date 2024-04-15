from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3
import os

app = Flask(__name__)

house_model_path = os.path.join('models_pickle', 'xgboost_model.pkl')
campus_model_path = os.path.join('models_pickle', 'llr.pkl')

# Load models
house_model = pickle.load(open(house_model_path, 'rb'))
campus_model = pickle.load(open(campus_model_path, 'rb'))

# welcome page
@app.route("/")
def welcome():
    return render_template("welcome.html")

#landing page for campus
@app.route("/campus/")
def campus_index():
    return render_template("campus/index_links.html")

# about page
@app.route("/about")
def about():
    return render_template("about.html")


# campus prediction
@app.route("/campus/predict", methods=['POST'])
def campus_predict():
   
    gender = int(request.form['gender'])
    ssc_p = float(request.form['ssc_p'])
    ssc_b = int(request.form['ssc_b'])
    hsc_p = float(request.form['hsc_p'])
    hsc_b = int(request.form['hsc_b'])
    hsc_s = int(request.form['hsc_s'])
    degree_p = float(request.form['degree_p'])
    degree_t = int(request.form['degree_t'])
    workex = int(request.form['workex'])
    etest_p = float(request.form['etest_p'])
    specialisation = int(request.form['specialisation'])
    mba_p = float(request.form['mba_p'])

    new_data = np.array([[gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p]])
    
    prediction = campus_model.predict_proba(new_data)[:, 1]  # Probability of positive class

    return render_template('campus/after.html', prediction=prediction)

#landing page for house
@app.route("/house/")
def house_index():
    return render_template("house/index_links_h.html")


#housing prediction
@app.route("/house/predict", methods=['POST'])
def house_predict():

    bedrooms = int(request.form['BHK'])
    size = int(request.form['Size'])
    floor = float(request.form['Floor'])
    area_type = float(request.form['Area Type'])
    
    city_name = request.form['City']
    furnishing_status = float(request.form['Furnishing Status'])
    tenant_preferred = float(request.form['Tenant Preferred'])
    bathroom = int(request.form['Bathroom'])
    point_of_contact = float(request.form['Point of Contact'])

    city_encoding = {
        'Kolkata': 11645.17366412,
        'Mumbai': 85321.20473251,
        'Bangalore': 24966.36568849,
        'Delhi': 29461.98347107,
        'Chennai': 21614.09203143,
        'Hyderabad': 20555.0483871
    }
    city = city_encoding.get(city_name, 0)

    new_data = np.array([[bedrooms, size, floor, area_type, city, 
                          furnishing_status, tenant_preferred, bathroom, point_of_contact]])
    prediction = house_model.predict(new_data)
    return render_template('house/after_h.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5003)

