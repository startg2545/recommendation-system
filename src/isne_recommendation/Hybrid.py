import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from datetime import datetime

def get_recommendations(username, i_data, ui_data, top_n):
    # Precompute KNN and TF-IDF matrices
    knn_matrix = getKNNMatrix(ui_data)
    tfidf_matrix = getTFIDFMatrix(i_data, ui_data)
    
    tfidf_weight = 0.3
    knn_weight = 0.7

    tfidf_matrix_normalized = tfidf_matrix * tfidf_weight
    knn_matrix_normalized = normalize(knn_matrix) * knn_weight

    combined_matrix = hstack((tfidf_matrix_normalized, knn_matrix_normalized))

    model_combined = NearestNeighbors(metric='cosine', algorithm='brute').fit(combined_matrix)
    
    # Get user's selected courses
    user_course = ui_data[ui_data['username'] == username]['course'].unique()

    def recommender_hybrid_all_courses(selected_courses, model_combined):
        courses = ui_data['course'].drop_duplicates()
        courses = courses.sort_values().set_axis(range(0,len(courses)))
        indices = [process.extractOne(course, courses)[2] for course in selected_courses]
        distances, indices = model_combined.kneighbors(combined_matrix[indices], n_neighbors=len(courses))
        recommendations = [courses[i].where(i!=idx) for idx, i in zip(indices, indices)]
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
    
    # Compute recommendations for all selected courses at once
    recommendations = recommender_hybrid_all_courses(user_course, model_combined)
    
    return recommendations.head(top_n)

def getKNNMatrix(df):
    courses = df['course'].drop_duplicates()
    courses = courses.sort_values().set_axis(range(0,len(courses)))
    user = pd.Series(df['username'])
    course = pd.Series(df['course'])
    def getEmailScore():
        emails = pd.Series(df['email'])
        email_score = []
        for data in emails:
            if data != '':
                if data == 'cmu.ac.th':
                    email_score.append(2)
                else:
                    email_score.append(1)
            else:
                email_score.append(0)
        return pd.Series(email_score)
    def getAgeEducationScore():
        ages = pd.Series(df['age'])
        educations = pd.Series(df['education'])
        def getAgeEducationScore(age, limit_age):
            if age <= limit_age:
                score = 0
            elif age <= limit_age + 2:
                score = 1
            else:
                score = 2
            return score
        set_nan = {np.nan}
        set_primary = {'Primary school level'}
        set_middleschool = {'Middle school level'}
        set_highschool = {'High school level', 'Vocational degree'}
        set_bachelor = {'Bachelor degree', 'High vocational degree', 'Diploma degree'}
        set_masterdoctor = {'Master degree', 'Ph.D.', 'Doctoral degree'}
        list_degree = ((set_nan, 0)), (set_primary, 15), (set_middleschool, 19), (set_highschool, 22), (set_bachelor,26), (set_masterdoctor,40)
        age_education_scores = []
        for i,x in enumerate(educations):
            count = False
            for y in list_degree:
                if x in y[0]:
                    age_education_scores.append(getAgeEducationScore(ages[i], y[1]))
                    count = True
            if count == False:
                print("User", i+2, " with education ", x, " is not in the list")
        return pd.Series(age_education_scores)
    def getPaymentScore():
        statuses = pd.Series(df['payment'])
        payment_score = []
        for data in statuses:
            if data == 'success':
                payment_score.append(2)
            if data == 'disapproval':
                payment_score.append(1)
            if data == 'failure':
                payment_score.append(0)
        return payment_score
    def getAddressScore():
        addresses = pd.Series(df['address'])
        address_score = []
        for data in addresses:
            if data == '':
                address_score.append(0)
            else:
                address_score.append(1)
        return pd.Series(address_score)
    def getTimeScore():
        date = pd.Series(df['time'])
        time_score = []
        date = pd.Series([datetime.strptime(info, '%Y-%m-%d %H:%M:%S') if type(info) != datetime else info for info in date])
        hour = pd.Series([info.hour for info in date], name='hour')
        time_set = {8,9,10,15}
        for info in hour:
            if info in time_set:
                time_score.append(1)
            else:
                time_score.append(0)
        return pd.Series(time_score)
    score = pd.Series( 1 + getEmailScore() * 0.375 + getAgeEducationScore() * 0.25 + getTimeScore() * 0.5 + getPaymentScore() * 1 + getAddressScore() * 0.25 )
    predata = pd.DataFrame({
        'User': user,
        'Course': course,
        'Score': score
    })
    user_course_tbl = pd.pivot_table(predata, values='Score', index='Course', columns='User').fillna(0)
    matrix = csr_matrix(user_course_tbl.values)
    return matrix

def getTFIDFMatrix(i_data, ui_data):
    selected_courses = ui_data['course'].drop_duplicates()
    content = i_data['course']
    content = content[content.isin(selected_courses)].drop_duplicates()
    courses = content.sort_values()
    courses = courses.set_axis(range(0, len(courses)))
    indexes = courses.index
    description = i_data['description'].iloc[indexes]
    description = description.set_axis(range(0, len(description)))
    matrix = TfidfVectorizer().fit_transform(description)
    return matrix

def get_recommendations(username, i_data, ui_data, top_n):
    
    
    tfidf_weight = 0.3
    knn_weight = 0.7
    
    tfidf_matrix = getTFIDFMatrix(i_data, ui_data)
    knn_matrix = getKNNMatrix(ui_data)

    tfidf_matrix_normalized = tfidf_matrix * tfidf_weight
    knn_matrix_normalized = normalize(knn_matrix) * knn_weight

    combined_matrix = hstack((tfidf_matrix_normalized, knn_matrix_normalized))

    model_combined = NearestNeighbors(metric='cosine', algorithm='brute').fit(combined_matrix)

    def recommender_hybrid_all_courses(course_name):
        courses = ui_data['course'].drop_duplicates()
        courses = courses.sort_values().set_axis(range(0,len(courses)))
        idx = process.extractOne(course_name, courses)[2]
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
    
    def recommender_hybrid_by_user(user_name):
        df = {
            'User': pd.Series(ui_data['username']),
            'Course': pd.Series(ui_data['course'])
        }
        user_course = pd.DataFrame(df)
        selected_user_name = user_course.loc[user_course['User'] == user_name]
        selected_courses = selected_user_name['Course']
        recommended_courses = [ recommender_hybrid_all_courses(x) for x in selected_courses]
        recommended_courses = [x[~x['Course'].isin(selected_courses)] for x in recommended_courses]
        df = pd.DataFrame({
            'Course': [],
            'Score': []
        }).rename_axis('Index')
        for x in recommended_courses:
            df = df._append(x)
        df =  df.sort_values('Score', ascending=False).drop_duplicates('Course')
        return df.head(top_n)
    
    return recommender_hybrid_by_user(username)