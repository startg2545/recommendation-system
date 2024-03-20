from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime
import pandas as pd
import numpy as np

def _get_feature_matrix(ui_data):
    ui_data = ui_data.set_axis(range(len(ui_data)))
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
    def getScore(age, limit_age):
        if age <= limit_age:
            score = 0
        elif age <= limit_age + 2:
            score = 1
        else:
            score = 2
        return score

    def getAgeEducationScore():
        ages = pd.Series(ui_data['age'])
        educations = pd.Series(ui_data['education'])
        set_nan = {np.nan}
        set_primary = {'Primary school level'}
        set_middleschool = {'Middle school level'}
        set_highschool = {'High school level', 'Vocational degree'}
        set_bachelor = {'Bachelor degree', 'High vocational degree', 'Diploma degree'}
        set_masterdoctor = {'Master degree', 'Ph.D.', 'Doctoral degree'}
        list_degree = ((set_nan, 0)), (set_primary, 15), (set_middleschool, 19), (set_highschool, 22), (set_bachelor,26), (set_masterdoctor,40)
        age_education_scores = []
        for i,x in enumerate(educations):
            for y in list_degree:
                if x in y[0]:
                    age_education_scores.append(getScore(ages[i], y[1]))
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
        return pd.Series(payment_score)
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

def train_test_split(ui_data, test_size):
    # Drop the rows that the same course has been taken by the same user
    drop_dup = ui_data.drop_duplicates(subset=['username', 'course'])

    # Drop the rows that users have taken only one course
    multiple_data = drop_dup[drop_dup['username'].map(drop_dup['username'].value_counts()) > 1]

    # Calculate the number of courses each user has taken
    courses_per_user = multiple_data.groupby('username')['course'].count().reset_index()
    courses_per_user.columns = ['username', 'course_count']

    # Merge the courses_per_user DataFrame back to the original DataFrame
    merged_user_courses = multiple_data.merge(courses_per_user, on='username')

    # Sort the DataFrame by username and course to ensure consistent train-test split
    user_courses = merged_user_courses.sort_values(by=['username', 'course'])

    # Initialize a counter variable to keep track of the number of courses for each user
    course_counter = 0
    current_user = None

    # Create a list to store the split information (True for training, False for testing)
    split_list = []

    # Iterate through each row to determine the split
    for index, row in user_courses.iterrows():
        if row['username'] != current_user:
            # We're at a new user, so reset the counter
            course_counter = 0
            current_user = row['username']

        if course_counter < ( ( 1 - test_size ) * row['course_count'] - 1 ):
            split_list.append(True)  # Training data
        else:
            split_list.append(False)  # Testing data

        course_counter += 1

    # Add the split information to the DataFrame
    user_courses['split'] = split_list

    # Split the DataFrame into training and testing sets based on the 'split' column
    train_ui_data = user_courses[user_courses['split']]
    test_ui_data = user_courses[~user_courses['split']]

    # Drop the auxiliary columns used for splitting
    train_ui_data = train_ui_data.drop(['course_count', 'split'], axis=1)
    test_ui_data = test_ui_data.drop(['course_count', 'split'], axis=1)

    return train_ui_data, test_ui_data


def hit_rate(train, test, model, k=10):
    # The names of users who have taken more than one course
    usernames = train['username'].value_counts()
    usernames = usernames[usernames > 1].index

    hits = []
    for name in usernames:
        predictions = predict(name, train, model, k)
        if type(predictions) is str:
            isHit = False
            hits.append(isHit)
        else:
            predictions = predictions['Course'].tolist()
            results = test[test['username'] == name]['course']
            test_predictions = [ result in predictions for result in results]
            isHit = [ True if pred == True else False for pred in test_predictions][0]
            hits.append(isHit)

    hits = np.count_nonzero(hits)
    accuracy = hits / len(usernames)
    
    return accuracy

def f1_score(train, test, model, k=10):
    # The names of users who have taken more than one course
    usernames = train['username'].value_counts()
    usernames = usernames[usernames > 1].index

    hits_of_users = []
    for name in usernames:
        predictions = predict(name, train, model, k)
        if type(predictions) is str:
            isHit = False
            hits_of_users.append(isHit)
        else:
            predictions = predictions['Course'].tolist()
            results = test[test['username'] == name]['course']
            test_predictions = [ result in predictions for result in results]
            isHit = [ True if pred == True else False for pred in test_predictions]
            hits_of_users.append(isHit)

    f1_list = []

    for hits in hits_of_users:
        
        # Calculate the parameters
        total_n_relevant_items = len(hits)
        n_relevant_items = np.count_nonzero(hits)
        
        # Recall @K
        recall = np.array([n_relevant_items / total_n_relevant_items])

        # Precision @K
        precision = np.array([n_relevant_items / k])

        # F1 Score
        if recall and precision == 0:
            f1 = 0
        else:
            dividend = 2 * precision * recall
            divisor = precision + recall
            with np.errstate(divide='ignore', invalid='ignore'):
                f1 = np.where(divisor != 0, np.divide( dividend, divisor ), 0)[0]
        
        f1_list.append(f1)
    
    accuracy = np.mean(f1_list)
        
    return accuracy