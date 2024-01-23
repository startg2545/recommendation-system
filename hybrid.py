#!/usr/bin/env python
# coding: utf-8

# Import required packages

# In[1]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import sys


# Check the validation of dataset format

# In[2]:

# Receive the user name from app.py
user_input = sys.argv[1]

# Get filename from pickle
with open('./pickle/filename.pickle', 'rb') as f:
    filename = pickle.load(f)

file_name = './uploads/' + filename
# file_name = './uploads/' + 'Sample Dataset.xlsx'
file = pd.read_excel(file_name)


# Count the number of courses

# In[3]:


course_counts = pd.DataFrame(file)['course'].value_counts()
courses = pd.Series(course_counts.index)
courses = courses.sort_values().set_axis(range(0,len(courses)))
number_of_courses = len(course_counts)


# Load matrices

# In[4]:


folder_path = './pickle'

with open(f'{folder_path}/knn_matrix.pickle', 'rb') as f:
    knn_matrix = pickle.load(f)

with open(f'{folder_path}/tfidf_matrix.pickle', 'rb') as f:
    tfidf_matrix = pickle.load(f)


# Combine knn_matrix and tfidf_matrix together

# In[5]:


from sklearn.preprocessing import normalize
from scipy.sparse import hstack

# Set the weight
tfidf_weight = 0.3
knn_weight = 0.7

# Normalize the matrices
tfidf_matrix_normalized = normalize(tfidf_matrix)
knn_matrix_normalized = normalize(knn_matrix)

# Combined the normalized metrices
combined_matrix = hstack([tfidf_matrix_normalized, knn_matrix_normalized])


# Euclidean Distance & Cosine Similarity

# In[6]:


model_combined = NearestNeighbors(metric='cosine', algorithm='brute').fit(combined_matrix)
'''This will create a new matrix where each row is the concatenation of the 
corresponding rows from tfidf_matrix and knn_matrix. The NearestNeighbors model
is then fit on this combined matrix.'''


# Recommender function using hybrid model

# In[7]:


# def recommender_hybrid(course_name):
#     n_recommendations = int( number_of_courses ** (1/2) )
#     idx = process.extractOne(course_name, courses)[2]
#     distances, indices = model_combined.kneighbors(combined_matrix[idx], n_neighbors=n_recommendations+1, return_distance=True)
#     recommendations = [courses[i].where(i!=idx) for i in indices]
#     recommended_courses = recommendations[0][1:]
#     course_distances = distances[0][1:]
#     d = {
#         'Course': recommended_courses,
#         'Cosine Distance': course_distances
#     }
#     results = pd.DataFrame(data=d)
#     n_distance = results['Cosine Distance'].where(results['Cosine Distance'] < 0.8).count()
#     return results.head(n_distance)

# KNN Recommender function for all courses

# In[8]:


# def recommender_knn_all_courses(course_name):
#     idx = process.extractOne(course_name, courses)[2]
#     # print('Selected movie:', courses[idx], 'Index:', idx)
#     distances, indices = model_combined.kneighbors(knn_matrix[idx], n_neighbors=len(courses))
#     recommendations = [courses[i].where(i!=idx) for i in indices]
#     recommended_courses = recommendations[0][1:]
#     scores = 1 - distances
#     course_distances = scores[0][1:]
#     d = {
#         'Course': recommended_courses,
#         'Score': course_distances
#     }
#     results = pd.DataFrame(data=d)
#     results = results.sort_index().rename_axis('Index')
#     return results


# KNN Recommender function using username

# In[9]:


# def recommender_knn_by_user(user_name, n_recommendations):
#     df = {
#         'User': pd.Series(file['username']),
#         'Course': pd.Series(file['course'])
#     }
    
#     user_course = pd.DataFrame(df)
#     selected_user_name = user_course.loc[user_course['User'] == user_name]
#     selected_courses = selected_user_name['Course']
    
#     recommended_courses = [ recommender_knn_all_courses(x) for x in selected_courses]
    
#     # pre dataframe
#     df = pd.DataFrame({
#         'Course': [],
#         'Score': []
#     }).rename_axis('Index')
    
#     for x in recommended_courses:
#         df = df._append(x)
#     df =  df.sort_values('Score', ascending=False).drop_duplicates('Course')
#     return df.head(n_recommendations)


# Hybrid Recommender function for all courses

# In[10]:


def recommender_hybrid_all_courses(course_name):
    idx = process.extractOne(course_name, courses)[2]
    # print('Selected movie:', courses[idx], 'Index:', idx)
    distances, indices = model_combined.kneighbors(combined_matrix[idx], n_neighbors=len(courses))
    recommendations = [courses[i].where(i!=idx) for i in indices]
    recommended_courses = recommendations[0][1:]
    scores = 1 - distances
    course_distances = scores[0][1:]
    d = {
        'Course': recommended_courses,
        'Score': course_distances
    }
    results = pd.DataFrame(data=d)
    results = results.sort_index().rename_axis('Index')
    return results


# Hybrid Recommender function using username

# In[11]:


def recommender_hybrid_by_user(user_name):

    n_recommendations = number_of_courses - 1

    df = {
        'User': pd.Series(file['username']),
        'Course': pd.Series(file['course'])
    }
    
    user_course = pd.DataFrame(df)
    selected_user_name = user_course.loc[user_course['User'] == user_name]
    selected_courses = selected_user_name['Course']
    
    recommended_courses = [ recommender_hybrid_all_courses(x) for x in selected_courses]
    
    # pre dataframe
    df = pd.DataFrame({
        'Course': [],
        'Score': []
    }).rename_axis('Index')
    
    for x in recommended_courses:
        df = df._append(x)
    df =  df.sort_values('Score', ascending=False).drop_duplicates('Course')
    return df.head(n_recommendations)


# Items show permanence whereas, people change with time
# Items are fewer in numbers to deal with. Which leads to smaller similarity matrix. Amazon and Netflix use it!
# Better for New users:
# — Him selecting just one item will let us provide recommendations
# — But for user based, new user has to wait until next build of similarity matrix (which is the only computational part of the framework)

# In[13]:


print(recommender_hybrid_by_user(user_input).to_html(index=False))


# Training and Testing part

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# k nearest neighbors

# Predata of training

# In[15]:


with open(f'{folder_path}/knn_X.pickle', 'rb') as f:
    knn_X = pickle.load(f)

with open(f'{folder_path}/knn_y.pickle', 'rb') as f:
    knn_y = pickle.load(f)


# Split the train and the test

# In[16]:


# Use the train test split function
X_train, X_test, y_train, y_test = train_test_split(
    knn_X, knn_y, random_state=42, test_size=0.25
)


# In[17]:


regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


# In[18]:


# print("Mean Squared Error:", mean_squared_error(predictions,y_test) / 4)
# print("Mean Absolute Error:", mean_absolute_error(predictions,y_test) / 4)


# Term Frequency and Inverse Document Frequency

# Predata of training

# In[19]:


with open(f'{folder_path}/tfidf_X.pickle', 'rb') as f:
    tfidf_X = pickle.load(f)
with open(f'{folder_path}/tfidf_y.pickle', 'rb') as f:
    tfidf_y = pickle.load(f)


# Split the train and the test

# In[20]:


# Split the train test data
X_train, X_test, y_train, y_test = train_test_split(tfidf_X, tfidf_y, test_size=0.2, random_state=42)

# Fit model selection
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
predictions = naive_bayes.predict(X_test)

# Convert the output to a boolean number
predictions = [ 1 if x == 'recommended' else 0 for x in predictions]
y_test = [ 1 if x == 'recommended' else 0 for x in y_test]


# Measure the performance for our parameters

# In[21]:


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# print(f'Accuracy: {accuracy} \nPrecision: {precision}, \nRecall: {recall}, \nF1: {f1}')


# %%
