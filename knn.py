#!/usr/bin/env python
# coding: utf-8

# Import required packages

# In[1]:

# In[1]:


import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import sys
import numpy as np
from datetime import datetime


# In[2]:

# Receive the user name from app.py<br>

# In[2]:


user_input = 'Martha Long'


# Get ui_dataset from pickle

# In[3]:


with open('./pickle/ui_dataset.pickle', 'rb') as f:
    ui_dataset = pickle.load(f)


# In[4]:


ui_dataset_file_name = './uploads/' + ui_dataset
my_user_item = pd.read_excel(ui_dataset_file_name)


# Count the number of courses

# In[3]:

# In[5]:


course_counts = pd.DataFrame(my_user_item)['course'].value_counts()
courses = pd.Series(course_counts.index)
courses = courses.sort_values().set_axis(range(0,len(courses)))
number_of_courses = len(course_counts)

# In[5]:

# Create path

# In[4]:

# In[6]:


import os
# Specify the folder path
folder_path = '/workspaces/recommendation-system/pickle'


# Create the folder if it doesn't exist

# In[7]:


if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# Create Series of users

# In[5]:

# In[8]:


s_name = my_user_item.loc[:, 'username']
Users = pd.Series(s_name, name='User')
Users


# Create Series of emails

# In[6]:

# In[9]:


s_email = my_user_item.loc[:, 'email'].fillna("")
Emails = pd.Series(s_email ,name='Email')
Emails


# Provide a score to each user based on their email domain

# In[7]:

# In[10]:


email_score = []
for data in Emails:
    if data != '':
        if data == 'cmu.ac.th':
            email_score.append(2)
        else:
            email_score.append(1)
    else:
        email_score.append(0)
email_score = pd.Series(email_score, name='Score Email')


# In[11]:


email_score


# Email Score Statistic

# In[8]:

# In[12]:


zero_score_count = email_score.where(email_score == 0).count()
one_score_count = email_score.where(email_score == 1).count()
two_score_count = email_score.where(email_score == 2).count()


# print("Number of students who filled cmu email:", two_score_count)<br>
# print("Number of students who filled other email:", one_score_count)<br>
# print("Number of students who did not fill email:", zero_score_count)

# Create function to calculate age-education score

# In[9]:

# In[13]:


def getAgeEducationScore(age, limit_age):
    if age <= limit_age:
        score = 0
    elif age <= limit_age + 2:
        score = 1
    else:
        score = 2
    return score


# Create set of the educational range

# In[10]:

# set_nan = {'อื่นๆ (-)', np.nan}<br>
# set_primaryschool = {'ประถมศึกษา', 'อื่นๆ (ป.4)', 'อื่นๆ (ป.7)', 'อื่นๆ (ป7)'}<br>
# set_middleschool = {'มัธยมศึกษาตอนต้น', 'Secondary school', 'อื่นๆ (มศ.3)'}<br>
# set_highschool = {'มัธยมศึกษาตอนปลาย', 'High school', 'Vocational', 'การศึกษานอกระบบ', <br>
#                   'ประกาศนียบัตรวิชาชีพ (ปวช.)', 'อื่นๆ (ม.ปลาย จบหลักสูตรEMR เป็นจนท.ปฏิบัติการ)',<br>
#                   'อื่นๆ (กำลังศึกษาชั้นมัธยมศึกษาตอนปลาย)', 'อื่นๆ (กำลังศึกษาชั้นมัธยมศึกษาปีที่6)', <br>
#                   'อื่นๆ (มศ.5)'}<br>
# set_bachelor = {'ปริญญาตรี', 'Bachelor degree', 'Diploma', 'High Vocational', <br>
#                 'ประกาศนียบัตรวิชาชีพชั้นสูง (ปวส.)', 'อื่นๆ (กำลังศึกษาในระดับปริญญาตรี)', <br>
#                 'อื่นๆ (กำลังศึกษาปริญญาตรี สาขารังสีเทคนิค)', 'อื่นๆ (ปริญญาแพทยศาสตร์บัณฑิต)', <br>
#                 'อื่นๆ (นักศึกษาแพทย์ปี 5)', 'อื่นๆ (นักศึกษาแพทย์ มช ปี4 ศูนย์เชียงราย)', <br>
#                 'อื่นๆ (แพทยศาสตร์บัณฑิต)', 'อื่นๆ (แพทย์)', 'อื่นๆ (ประกาศณียบัตรผู้ช่วยพยาบาล)', <br>
#                 'อนุปริญญา', 'อื่นๆ (ป.ตรี)', 'อื่นๆ (ผู้ช่วยพยาบาล)'}<br>
# set_masterdoctor = {'ปริญญาโท', 'ปริญญาเอก', "Master's degree", 'Other (OBGYN specalist lavel 1)', <br>
#                     'Other (Residency)', 'Ph.D.', 'อื่นๆ (Internal Medicine)', <br>
#                     'อื่นๆ (เฉพาะทาง)', 'อื่นๆ (วุฒิบัตร)', 'อื่นๆ (วว.ออร์โธปิดิกส์)', <br>
#                     'อื่นๆ (วุฒิบัตรแสดงความรู้ความชำนาญในการประกอบวิชาชีพเภสัชกรรม สาขาเภสัชบำบัด)', <br>
#                     'อื่นๆ (วุฒิบัตรผู้เชี่ยวชาญสาขาทันตกรรมทั่วไป)', 'อื่นๆ (วุฒิบัตรศัลยศาสตร์และแม็กซิลโลเฟเชียล)'}

# In[14]:


set_nan = {np.nan}
set_primary = {'Primary school level'}
set_middleschool = {'Middle school level'}
set_highschool = {'High school level', 'Vocational degree'}
set_bachelor = {'Bachelor degree', 'High vocational degree', 'Diploma degree'}
set_masterdoctor = {'Master degree', 'Ph.D.', 'Doctoral degree'}


# In[15]:


list_degree = ((set_nan, 0)), (set_primary, 15), (set_middleschool, 19), (set_highschool, 22), (set_bachelor,26), (set_masterdoctor,40)


# Create Series of Age-Education

# In[11]:

# In[16]:


ages = my_user_item.loc[:, 'age']
educations = my_user_item.loc[:, 'education']
age_education_scores = []


# In[17]:


for i,x in enumerate(educations):
    count = False
    for y in list_degree:
        if x in y[0]:
            age_education_scores.append(getAgeEducationScore(ages[i], y[1]))
            count = True
    if count == False:
        print("User", i + 2, " with education", x, "is not in the list")


# In[18]:


age_education_scores = pd.Series(age_education_scores, name='Age Education Score')


# Age-Education Score Statistic

# In[12]:

# In[19]:


zero_score_count = age_education_scores.where(age_education_scores == 0).count()
one_score_count = age_education_scores.where(age_education_scores == 1).count()
two_score_count = age_education_scores.where(age_education_scores == 2).count()


# print("Number of students who are currently in the educational system:", zero_score_count)<br>
# print("Number of students who were recently graduated:", one_score_count)<br>
# print("Number of students who are not in the educational system:", two_score_count)

# Create Series of status

# In[13]:

# In[20]:


status = my_user_item.loc[:, 'payment'].fillna("")
status = pd.Series(status ,name='Status')
status


# Provide a score to each user based on their purchase status

# In[14]:

# In[21]:


payment_score = []
for x in status:
    if x == 'success':
        payment_score.append(2)
    if x == 'disapproval':
        payment_score.append(1)
    if x == 'failure':
        payment_score.append(0)
payment_score = pd.Series(payment_score)


# Purchase Status Score Statistics

# In[15]:

# In[22]:


zero_score_count = payment_score.where(payment_score == 0).count()
one_score_count = payment_score.where(payment_score == 1).count()
two_score_count = payment_score.where(payment_score == 2).count()


# print("Number of students who are in arrears:", zero_score_count)<br>
# print("Number of students whose payment was not approved:", one_score_count)<br>
# print("Number of students with payment approval:", two_score_count)

# Create Series of address

# In[16]:

# In[23]:


address = my_user_item.loc[:, 'address'].fillna("")
address = pd.Series(address ,name='Status')
address


# Provide a score to each user based on whether they provide address information or not

# In[17]:

# In[24]:


address_score = [ 0 if x == '' else 1 for x in address]
address_score = pd.Series(address_score)


# Address Score Statistic

# In[18]:

# In[25]:


zero_score_count = address_score.where(address_score == 0).count()
one_score_count = address_score.where(address_score == 1).count()


# print("Number of students who did not fill address:", zero_score_count)<br>
# print("Number of students who filled address:", one_score_count)

# Create Series of data

# In[19]:

# In[54]:


date = my_user_item.loc[:, 'time']
# Convert string to datetime object
date = pd.Series([datetime.strptime(info, '%Y-%m-%d %H:%M:%S') if type(info) != datetime else info for info in date])
hour = pd.Series([info.hour for info in date], name='hour')
# Provide a score to each user based on their enrollment time

# In[20]:

# In[57]:


time_set = {8,9,10,15}  # if the time is in this set, it is considered as a good time
time_score = [ 1 if x in time_set else 0 for x in hour]
time_score = pd.Series(time_score)


# Create DataFrame by merging these 4 Series and calculate impressive level

# In[21]:

# In[58]:


user = my_user_item.loc[:, 'username']
course = my_user_item.loc[:, 'course']
d = {
    'User': user,
    'Course': course,
    'Email Score': email_score,
    'Age Education Score': age_education_scores,
    'Time': time_score,
    'Payment Score': payment_score,
    'Address Score': address_score,
    'Score': 1 + email_score*0.375 + age_education_scores*0.25 + time_score*0.5  + payment_score*1 + address_score*0.25
}
knn_features = pd.DataFrame(d)
knn_features


# Create user-course table

# In[22]:

# all user, course, score have the same length

# In[59]:


data = {
    'User': knn_features['User'],
    'Course': knn_features['Course'],
    'Score': knn_features['Score'],
}


# In[60]:


predata = pd.DataFrame(data)


# Calculate sparsity and csr matrix

# In[23]:

# Pivot table by rotating course

# In[61]:


data = pd.pivot_table(predata, values='Score', index='Course', columns='User').fillna(0)


# Convert dataframe to sparse matrix

# In[62]:


knn_matrix = csr_matrix(data.values)

data
# Export file

# In[24]:

# In[63]:


file_path = os.path.join(folder_path, 'knn_matrix.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(knn_matrix, f)


# Create model

# In[25]:

# In[64]:


model_knn = NearestNeighbors(metric='cosine', algorithm='brute').fit(knn_matrix)


# Recommender function using KNN model

# In[26]:

# In[65]:


def recommender_knn(course_name):
    n_recommendations = int( number_of_courses ** (1/2) ) + 1
    idx = process.extractOne(course_name, courses)[2]
    distances, indices = model_knn.kneighbors(knn_matrix[idx], n_neighbors=n_recommendations, return_distance=True)
    return courses
    recommendations = [courses[i].where(i!=idx) for i in indices]
    recommended_courses = recommendations[0][1:]
    course_distances = distances[0][1:]
    d = {
        'Course': recommended_courses,
        'Cosine Distance': course_distances
    }
    results = pd.DataFrame(data=d)
    n_distance = results['Cosine Distance'].where(results['Cosine Distance'] < 0.8).count()
    return results.head(n_distance)


# In[27]:

# In[66]:


def recommender_knn_by_user(username):
    # Filter the courses that the user has already taken
    user_courses = my_user_item.loc[my_user_item['username'] == username]['course']

    return recommender_knn(user_courses[0])
    # Get the recommendations based on these courses
    recommendations = [recommender_knn(course) for course in user_courses]

    # Reset index
    recommendations = pd.concat(recommendations).drop_duplicates(subset='Course', keep='first')

    # Drop duplicated courses
    recommendations = recommendations.reset_index(drop=True)

    # Drop the courses that the user has already taken
    recommendations = recommendations[~recommendations['Course'].isin(user_courses)]
    # Add the new column called 'Score'
    recommendations['Score'] = recommendations['Cosine Distance'].apply(lambda x: 1 - x)
    return recommendations


# In[67]:

print(recommender_knn_by_user('Martha Long'))
# print(recommender_knn_by_user(user_input).to_html(index=False))

# Predata of training

# In[28]:

# Take 2 columns from Table

# In[ ]:


course_score = knn_features[['Course', 'Score']]


# Sort the course

# In[ ]:


course = knn_features['Course'].sort_values().unique()


# Calculate the mean of each course

# In[ ]:


course_mean = course_score.groupby('Course')
course_mean = course_mean.mean().loc[:, 'Score']


# Take 2 columns from Table

# In[ ]:


course_user = knn_features[['Course', 'User']]


# Count the number of users who enrolled each course 

# In[ ]:


course_count = course_user.value_counts('Course')
course_count = course_count.sort_index()


# Take feature columns from Table

# In[ ]:


course_feature = knn_features[['Course', 'Email Score', 'Age Education Score', 'Time', 'Payment Score', 'Address Score']]
agg_functions = {
    'Email Score': 'mean',
    'Age Education Score': 'mean',
    'Time': 'mean',
    'Payment Score': 'mean',
    'Address Score': 'mean',
}
course_feature = course_feature.groupby('Course').aggregate(agg_functions)


# Create the train dataframe

# In[29]:

# In[ ]:


course_feature['Count'] = course_count
course_feature['Score'] = course_mean
train_df = pd.DataFrame(course_feature)
pd.set_option('display.max_rows', train_df.shape[0]+1)
filtered_train_df = train_df[train_df['Score'] > 4]


# In[30]:

# Get X and y features variables

# In[ ]:


X = train_df[['Email Score', 'Age Education Score', 'Time', 'Payment Score', 'Address Score', 'Count']]
y = train_df['Score']


# Export file

# In[31]:

# In[ ]:


file_path = os.path.join(folder_path, 'knn_X.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(X, f)


# In[ ]:


file_path = os.path.join(folder_path, 'knn_y.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(y, f)


# %%
