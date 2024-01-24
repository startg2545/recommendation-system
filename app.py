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
        f = request.files['filename']
        upload_dir = './uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)  # create directory if it does not exist
        f.save(os.path.join(upload_dir, f.filename))
        session['filename'] = f.filename  # store filename in session
        return redirect(url_for('predata'))  # redirect to predata page
    return render_template('homepage.html')

@app.route('/predata')
def predata():
    filename = session.get('filename')  # get filename from session
    
    # Save filename to pickle
    folder_path = '/workspaces/recommendation-system/pickle'
    file_path = os.path.join(folder_path, 'filename.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(filename, f)
        
    # Render predata page
    return render_template('predata.html', filename=filename)

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