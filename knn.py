#!/usr/bin/env python
# coding: utf-8

# Import required packages

# In[1]:


import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


# In[2]:


file_name = 'dataset.xlsx'
file = pd.read_excel(file_name)


# Count the number of courses

# In[3]:


course_counts = pd.DataFrame(file)['course'].value_counts()
courses = pd.Series(course_counts.index)
courses = courses.sort_values().set_axis(range(0,len(courses)))
number_of_courses = len(course_counts)


# Create path

# In[4]:


import os
# Specify the folder path
folder_path = '/workspaces/recommendation-system/pickle'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# Create Series of users

# In[5]:


s_name = file.loc[:, 'username']
Users = pd.Series(s_name, name='User')
Users


# Create Series of emails

# In[6]:


s_email = file.loc[:, 'email'].fillna("")
Emails = pd.Series(s_email ,name='Email')
Emails


# Provide a score to each user based on their email domain

# In[7]:


email_score = []
for data in Emails:
    if data != '':
        if data.split('@')[1] == 'cmu.ac.th':
            email_score.append(2)
        else:
            email_score.append(1)
    else:
        email_score.append(0)
email_score = pd.Series(email_score, name='Score Email')

email_score


# Email Score Statistic

# In[8]:


zero_score_count = email_score.where(email_score == 0).count()
one_score_count = email_score.where(email_score == 1).count()
two_score_count = email_score.where(email_score == 2).count()

print("Number of students who filled cmu email:", two_score_count)
print("Number of students who filled other email:", one_score_count)
print("Number of students who did not fill email:", zero_score_count)


# Create function to calculate age-education score

# In[9]:


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


set_nan = {'อื่นๆ (-)', np.nan}
set_primaryschool = {'ประถมศึกษา', 'อื่นๆ (ป.4)', 'อื่นๆ (ป.7)', 'อื่นๆ (ป7)'}
set_middleschool = {'มัธยมศึกษาตอนต้น', 'Secondary school', 'อื่นๆ (มศ.3)'}
set_highschool = {'มัธยมศึกษาตอนปลาย', 'High school', 'Vocational', 'การศึกษานอกระบบ', 
                  'ประกาศนียบัตรวิชาชีพ (ปวช.)', 'อื่นๆ (ม.ปลาย จบหลักสูตรEMR เป็นจนท.ปฏิบัติการ)',
                  'อื่นๆ (กำลังศึกษาชั้นมัธยมศึกษาตอนปลาย)', 'อื่นๆ (กำลังศึกษาชั้นมัธยมศึกษาปีที่6)', 
                  'อื่นๆ (มศ.5)'}
set_bachelor = {'ปริญญาตรี', 'Bachelor degree', 'Diploma', 'High Vocational', 
                'ประกาศนียบัตรวิชาชีพชั้นสูง (ปวส.)', 'อื่นๆ (กำลังศึกษาในระดับปริญญาตรี)', 
                'อื่นๆ (กำลังศึกษาปริญญาตรี สาขารังสีเทคนิค)', 'อื่นๆ (ปริญญาแพทยศาสตร์บัณฑิต)', 
                'อื่นๆ (นักศึกษาแพทย์ปี 5)', 'อื่นๆ (นักศึกษาแพทย์ มช ปี4 ศูนย์เชียงราย)', 
                'อื่นๆ (แพทยศาสตร์บัณฑิต)', 'อื่นๆ (แพทย์)', 'อื่นๆ (ประกาศณียบัตรผู้ช่วยพยาบาล)', 
                'อนุปริญญา', 'อื่นๆ (ป.ตรี)', 'อื่นๆ (ผู้ช่วยพยาบาล)'}
set_masterdocter = {'ปริญญาโท', 'ปริญญาเอก', "Master's degree", 'Other (OBGYN specalist lavel 1)', 
                    'Other (Residency)', 'Ph.D.', 'อื่นๆ (Internal Medicine)', 
                    'อื่นๆ (เฉพาะทาง)', 'อื่นๆ (วุฒิบัตร)', 'อื่นๆ (วว.ออร์โธปิดิกส์)', 
                    'อื่นๆ (วุฒิบัตรแสดงความรู้ความชำนาญในการประกอบวิชาชีพเภสัชกรรม สาขาเภสัชบำบัด)', 
                    'อื่นๆ (วุฒิบัตรผู้เชี่ยวชาญสาขาทันตกรรมทั่วไป)', 'อื่นๆ (วุฒิบัตรศัลยศาสตร์และแม็กซิลโลเฟเชียล)'}

list_degree = ((set_nan, 0), (set_primaryschool, 16), (set_middleschool, 19), 
               (set_highschool, 22), (set_bachelor,26), (set_masterdocter,40))


# Create Series of Age-Education

# In[11]:


ages = file.loc[:, 'age']
educations = file.loc[:, 'education']
age_education_scores = []

for i,x in enumerate(educations):
    for y in list_degree:
        if x in y[0]:
            age_education_scores.append(getAgeEducationScore(ages[i], y[1]))
            
age_education_scores = pd.Series(age_education_scores, name='Age Education Score')


# Age-Education Score Statistic

# In[12]:


zero_score_count = age_education_scores.where(age_education_scores == 0).count()
one_score_count = age_education_scores.where(age_education_scores == 1).count()
two_score_count = age_education_scores.where(age_education_scores == 2).count()

print("Number of students who are currently in the educational system:", zero_score_count)
print("Number of students who were recently graduated:", one_score_count)
print("Number of students who are not in the educational system:", two_score_count)


# Create Series of status

# In[13]:


status = file.loc[:, 'payment'].fillna("")
status = pd.Series(status ,name='Status')
status


# Provide a score to each user based on their purchase status

# In[14]:


payment_score = []
for x in status:
    if x == 'ชำระเงิน':
        payment_score.append(2)
    if x == 'ไม่ผ่านการอนุมัติ':
        payment_score.append(1)
    if x == 'ค้างชำระ':
        payment_score.append(0)
payment_score = pd.Series(payment_score)


# Purchase Status Score Statistics

# In[15]:


zero_score_count = payment_score.where(payment_score == 0).count()
one_score_count = payment_score.where(payment_score == 1).count()
two_score_count = payment_score.where(payment_score == 2).count()

print("Number of students who are in arrears:", zero_score_count)
print("Number of students whose payment was not approved:", one_score_count)
print("Number of students with payment approval:", two_score_count)


# Create Series of address

# In[16]:


address = file.loc[:, 'address'].fillna("")
address = pd.Series(address ,name='Status')
address


# Provide a score to each user based on whether they provide address information or not

# In[17]:


address_score = [ 0 if x == '' else 1 for x in address]
address_score = pd.Series(address_score)


# Address Score Statistic

# In[18]:


zero_score_count = address_score.where(address_score == 0).count()
one_score_count = address_score.where(address_score == 1).count()

print("Number of students who did not fill address:", zero_score_count)
print("Number of students who filled address:", one_score_count)


# Create Series of data

# In[19]:


time = file.loc[:, 'time'].fillna("")
time = pd.Series(time ,name='Time')
time = time.str.slice(start=-8, stop=-6)


# Provide a score to each user based on their enrollment time

# In[20]:


time_set = {'08','09','10','15'}  # if the time is in this set, it is considered as a good time
time_score = [ 1 if x in time_set else 0 for x in time]
time_score = pd.Series(time_score)


# Create DataFrame by merging these 4 Series and calculate impressive level

# In[21]:


user = file.loc[:, 'username']
course = file.loc[:, 'course']
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
knn_features.head(20)


# Create user-course table

# In[22]:


# all user, course, score have the same length
data = {
    'User': knn_features['User'],
    'Course': knn_features['Course'],
    'Score': knn_features['Score'],
}

predata = pd.DataFrame(data)


# Calculate sparsity and csr matrix

# In[23]:


# Pivot table by rotating course
data = pd.pivot_table(predata, values='Score', index='Course', columns='User').fillna(0)

# Convert dataframe to sparse matrix
knn_matrix = csr_matrix(data.values)


# Export file

# In[24]:


file_path = os.path.join(folder_path, 'knn_matrix.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(knn_matrix, f)


# Create model

# In[25]:


model_knn = NearestNeighbors(metric='cosine', algorithm='brute').fit(knn_matrix)


# Recommender function using KNN model

# In[26]:


def recommender_knn(course_name):
    n_recommendations = int( number_of_courses ** (1/2) )
    idx = process.extractOne(course_name, courses)[2]
    distances, indices = model_knn.kneighbors(knn_matrix[idx], n_neighbors=n_recommendations+1, return_distance=True)
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


recommender_knn('หลักการและพื้นฐานของเครื่องมือทางรังสีวิทยา (Basic Principle of Diagnostic Radiology Imaging Instruments)')


# Predata of training

# In[28]:


# Take 2 columns from Table
course_score = knn_features[['Course', 'Score']]

# Sort the course
course = knn_features['Course'].sort_values().unique()

# Calculate the mean of each course
course_mean = course_score.groupby('Course')
course_mean = course_mean.mean().loc[:, 'Score']

# Take 2 columns from Table
course_user = knn_features[['Course', 'User']]

# Count the number of users who enrolled each course 
course_count = course_user.value_counts('Course')
course_count = course_count.sort_index()

# Take feature columns from Table
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


course_feature['Count'] = course_count
course_feature['Score'] = course_mean
train_df = pd.DataFrame(course_feature)
pd.set_option('display.max_rows', train_df.shape[0]+1)
filtered_train_df = train_df[train_df['Score'] > 4]


# In[30]:


# Get X and y features variables
X = train_df[['Email Score', 'Age Education Score', 'Time', 'Payment Score', 'Address Score', 'Count']]
y = train_df['Score']


# Export file

# In[31]:


file_path = os.path.join(folder_path, 'knn_X.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(X, f)

file_path = os.path.join(folder_path, 'knn_y.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(y, f)

