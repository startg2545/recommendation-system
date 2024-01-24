# Import required packages
import pandas as pd
import sys
import pickle

# Receive the user name from app.py
user_input = sys.argv[1]

# Get filename from pickle
with open('./pickle/filename.pickle', 'rb') as f:
    filename = pickle.load(f)

file_name = './uploads/' + filename
file = pd.read_excel(file_name)

# Get courses for selected user
selected_username = file[file['username'] == user_input]
selected_courses = selected_username['course']

# Print selected courses to html
print(selected_courses.to_numpy())