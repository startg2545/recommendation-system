from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime
import pandas as pd
import numpy as np

def _get_feature_matrix(ui_data):
    user = pd.Series(ui_data['username'])
    course = pd.Series(ui_data['course'])

    # Calculate the scores for the user-item dataframe
    def getEmailScore():
        emails = pd.Series(ui_data['email'])
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
        ages = pd.Series(ui_data['age'])
        educations = pd.Series(ui_data['education'])
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
        statuses = pd.Series(ui_data['payment'])
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
        addresses = pd.Series(ui_data['address'])
        address_score = []
        for data in addresses:
            if data == '':
                address_score.append(0)
            else:
                address_score.append(1)
        return pd.Series(address_score)
    def getTimeScore():
        date = pd.Series(ui_data['time'])
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

def fit(ui_data):
    matrix = _get_feature_matrix(ui_data)
    model = NearestNeighbors(metric='cosine', algorithm='brute').fit(matrix)
    return model

def predict(username, ui_data, model, top_n):
    # Take the necessary data from the user-item dataframe
    courses = ui_data['course'].drop_duplicates()
    courses = courses.sort_values().set_axis(range(0,len(courses)))

    # Take matrix from model
    matrix = _get_feature_matrix(ui_data)

    def get_recommendations_for_course(course):
        idx = process.extractOne(course, courses)[2]
        distances, indices = model.kneighbors(matrix[idx], n_neighbors=top_n+1, return_distance=True)
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

    # Filter the courses that the user has already taken
    user_courses = ui_data.loc[ui_data['username'] == username]['course']
    # Get the recommendations based on taken courses
    recommendations = [ get_recommendations_for_course(course) for course in user_courses ]
    
    if len(recommendations) == 0:
        return 'No recommendations available for this user.'

    # Reset index
    recommendations = pd.concat(recommendations).drop_duplicates(subset='Course', keep='first')

    # Drop duplicated courses
    recommendations = recommendations.reset_index(drop=True)

    # Drop the courses that the user has already taken
    recommendations = recommendations[~recommendations['Course'].isin(user_courses)]

    # Add the new column called 'Score'
    recommendations['Score'] = recommendations['Cosine Distance'].apply(lambda x: 1 - x)
    
    return recommendations

def train_test_split(ui_data):
    # Drop courses that have been taken more than 1 time by the same user
    ui_data_drop = ui_data.drop_duplicates(subset=['username', 'course'])

    # Calculate the number of courses each user has taken
    courses_per_user = ui_data_drop.groupby('username')['course'].count().reset_index()
    courses_per_user.columns = ['username', 'course_count']

    # Merge the courses_per_user DataFrame back to the original DataFrame
    merged_user_courses = ui_data_drop.merge(courses_per_user, on='username')

    # Sort the DataFrame by username and course to ensure consistent train-test split
    user_courses = merged_user_courses.sort_values(by=['username', 'course'])

    # Initialize a counter variable to keep track of the number of courses for each user
    course_counter = 1

    # Create a list to store the split information (True for training, False for testing)
    split_list = []

    # Iterate through each row to determine the split
    for index, row in user_courses.iterrows():
        if course_counter < row['course_count']:
            split_list.append(True)  # Training data
        else:
            split_list.append(False)  # Testing data
            course_counter = 1  # Reset the counter for the next user
        course_counter += 1

    # Add the split information to the DataFrame
    user_courses['split'] = split_list

    # Split the DataFrame into training and testing sets based on the 'split' column
    train_ui_data = user_courses[user_courses['split']]
    test_ui_data = user_courses[~user_courses['split']]

    # Drop the auxiliary columns used for splitting
    train_ui_data = train_ui_data.drop(['course_count', 'split'], axis=1)
    test_ui_data = test_ui_data.drop(['course_count', 'split'], axis=1)

    train_ui_data = train_ui_data.set_axis(range(len(train_ui_data)))
    test_ui_data = test_ui_data.set_axis(range(len(test_ui_data)))

    return train_ui_data, test_ui_data


def evaluate_model(train, test, model, top_n):
    # The names of users who have taken more than one course
    usernames = train['username'].value_counts()
    usernames = usernames[usernames > 1].index

    hit = []
    for name in usernames:
        predictions = predict(name, train, model, top_n)
        if type(predictions) is str:
            isHit = False
            hit.append(isHit)
        else:
            predictions = predictions['Course'].tolist()
            results = test[test['username'] == name]['course'].iloc[0]
            isHit = results in predictions
            hit.append(isHit)

    hits = np.count_nonzero(hit)
    accuracy = hits / len(usernames)
    
    return accuracy