#%% md
# Import required packages
#%%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import math
#%% md
# Read the file
#%%
file = pd.read_excel('shortcourses2566.xlsx')
#%% md
# Count the number of courses
#%%
course_counts = pd.DataFrame(file)['หลักสูตรอบรมระยะสั้น'].value_counts()
#%% md
# Create Series of users
#%%
s_name = file.loc[:, 'ชื่อ-นามสกุล (อังกฤษ)']
Users = pd.Series(s_name, name='Users')
Users
#%% md
# Create Series of emails
#%%
s_email = file.loc[:, 'อีเมล'].fillna("")
Emails = pd.Series(s_email ,name='Emails')
Emails
#%% md
# Provide a score to each user based on their email domain
#%%
email_score = []
for data in Emails:
    if data != '':
        if data.split('@')[1] == 'cmu.ac.th':
            email_score.append(2)
        else:
            email_score.append(1)
    else:
        email_score.append(0)
email_score = pd.Series(email_score, name='Score Emails')

email_score
#%% md
# Email Score Statistic
#%%

#%% md
# Create function to calculate age-education score
#%%
def getAgeEducationScore(age, limit_age):
    if age <= limit_age:
        score = 1
    elif limit_age == 0:
        score = 0
    else:
        score = 3
    return score
#%% md
# Create set of the educational range
#%%
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
#%% md
# Create Series of Age-Education
#%%
ages = file.loc[:, 'อายุ']
educations = file.loc[:, 'วุฒิการศึกษา']
age_education_scores = []

for i,x in enumerate(educations):
    for y in list_degree:
        if x in y[0]:
            age_education_scores.append(getAgeEducationScore(ages[i], y[1]))
            
age_education_scores = pd.Series(age_education_scores, name='Age Education Score')
#%% md
# Age-Education Score Statistic
#%%

#%% md
# Create Series of status
#%%
status = file.loc[:, 'สถานะ'].fillna("")
status = pd.Series(status ,name='status')
status
#%% md
# Provide a score to each user based on their purchase status
#%%
status_score = []
for x in status:
    if x == 'ชำระเงิน':
        status_score.append(8)
    if x == 'ไม่ผ่านการอนุมัติ':
        status_score.append(7)
    if x == 'ค้างชำระ':
        status_score.append(5)
status_score = pd.Series(status_score)
#%% md
# Purchase Status Score Statistics
#%%

#%% md
# Create Series of address
#%%
address = file.loc[:, 'ที่อยู่'].fillna("")
address = pd.Series(address ,name='status')
address
#%% md
# Provide a score to each user based on whether they provide address information or not
#%%
address_score = [ 1 if x == '' else 2 for x in address]
address_score = pd.Series(address_score)
#%% md
# Address Score Statistic
#%%

#%% md
# Convert list to pandas series
#%%
email_score = pd.Series(email_score)
age_education_scores = pd.Series(age_education_scores)
status_score = pd.Series(status_score)
address_score = pd.Series(address_score)
#%% md
# Create DataFrame by merging these 4 Series and calculate impressive level
#%%
d = {
    'Email Score': email_score,
    'Age Education Score': age_education_scores,
    'Payment Score': status_score,
    'Address Score': address_score,
    'Point': email_score + status_score + address_score + age_education_scores,
    'Impressive Level': ( email_score + status_score + address_score + age_education_scores ) / 17
}
df = pd.DataFrame(d)
df
#%% md
# Create user-course table
#%%
user = file.loc[:, 'ชื่อ-นามสกุล (อังกฤษ)']
course = file.loc[:, 'หลักสูตรอบรมระยะสั้น']
score = df['Impressive Level']
# all user, course, score have the same length
data = {
    'user': user,
    'course': course,
    'score': score,
}

predata = pd.DataFrame(data)
#%% md
# Calculate sparsity and csr matrix
#%%
data = predata.pivot_table(index='course', columns='user', values='score').fillna(0)
data_mtx = csr_matrix(data)
courses = pd.Series(data.index)
#%% md
# Euclidean Distance & Cosine Similarity
#%%
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=int(np.around(math.sqrt(len(courses))))).fit(data_mtx)
model_knn
#%%
def recommender_knn(course_name, n_recommendations):
    model_knn.fit(data_mtx)
    idx = process.extractOne(course_name, courses)[2]
    print('Selected movie:', courses[idx], 'Index:', idx)
    print('Searching for recommendations...')
    distances, indices = model_knn.kneighbors(data_mtx[idx], n_neighbors=n_recommendations+1)
    recommendations = [courses[i].where(i!=idx) for i in indices]
    recommended_courses = recommendations[0][1:]
    course_distances = distances[0][1:]
    d = {
        'recommended_courses': recommended_courses,
        'course_distances': course_distances
    }
    results = pd.DataFrame(data=d)
    return results