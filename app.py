import os
from flask import Flask, request, redirect, session, url_for, render_template
import pickle
import subprocess
import numpy as np

app = Flask(__name__)
app.secret_key = 'iqcejokmkbogg'  # replace with your secret key

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        ui_dataset = request.files['my-user-item']
        i_dataset = request.files['my-item']
        upload_dir = './uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)  # create directory if it does not exist
        ui_dataset.save(os.path.join(upload_dir, ui_dataset.filename))
        i_dataset.save(os.path.join(upload_dir, i_dataset.filename))
        session['my-user-item'] = ui_dataset.filename  # store filename in session
        session['my-item'] = i_dataset.filename
        return redirect(url_for('predata'))  # redirect to predata page
    return render_template('homepage.html')

@app.route('/predata')
def predata():
    ui_dataset = session.get('my-user-item')  # get my-user-item from session
    i_dataset = session.get('my-item')  # get my-item from session

    # Save filename to pickle
    folder_path = '/workspaces/recommendation-system/pickle'
    ui_dataset_path = os.path.join(folder_path, 'ui_dataset.pickle')
    i_dataset_path = os.path.join(folder_path, 'i_dataset.pickle')
    with open(ui_dataset_path, 'wb') as f:
        pickle.dump(ui_dataset, f)
    with open(i_dataset_path, 'wb') as f:
        pickle.dump(i_dataset, f)
        
    # Render predata page
    return render_template('predata.html', ui_dataset=ui_dataset, i_dataset=i_dataset)

@app.route('/get-recommendation', methods=['GET', 'POST'])
def get_recommendation():

    # Get courses that a selected user has taken
    user = request.form.get('user')
    result = subprocess.run(['python', 'get_courses.py', user], capture_output=True, text=True).stdout
    # Convert string to list
    courses = eval(result)

    # Get recommendations
    if 'tfidf' in request.form:
        model = 'tfidf'
        user = request.form.get('user')
        result = subprocess.run(['python', 'tfidf.py', user], capture_output=True, text=True)
        dataframe = result.stdout
    elif 'knn' in request.form:
        model = 'knn'
        user = request.form.get('user')
        result = subprocess.run(['python', 'knn.py', user], capture_output=True, text=True)
        dataframe = result.stdout
    elif 'hybrid' in request.form:
        model = 'hybrid'
        user = request.form.get('user')
        result = subprocess.run(['python', 'hybrid.py', user], capture_output=True, text=True)
        dataframe = result.stdout
    return render_template('recommendation.html', user=user, model=model, dataframe=dataframe, courses=courses)

if __name__ == '__main__':
    app.debug = True
    app.run()