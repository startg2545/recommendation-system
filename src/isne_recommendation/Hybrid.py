import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from datetime import datetime

def _get_feature_matrix(i_data, ui_data):

    def getKNNMatrix(ui_data):
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

    tfidf_weight = 0.3
    knn_weight = 0.7
    
    tfidf_matrix = getTFIDFMatrix(i_data, ui_data)
    knn_matrix = getKNNMatrix(ui_data)

    tfidf_matrix_normalized = tfidf_matrix * tfidf_weight
    knn_matrix_normalized = normalize(knn_matrix) * knn_weight

    combined_matrix = hstack((tfidf_matrix_normalized, knn_matrix_normalized))

    return combined_matrix

def fit(i_data, ui_data):
    matrix = _get_feature_matrix(i_data, ui_data)
    model_combined = NearestNeighbors(metric='cosine', algorithm='brute').fit(matrix)
    return model_combined

def predict(username, i_data, ui_data, model, top_n):
    combined_matrix = _get_feature_matrix(i_data, ui_data)

    def recommender_hybrid_all_courses(course_name):
        courses = ui_data['course'].drop_duplicates()
        courses = courses.sort_values().set_axis(range(0,len(courses)))
        idx = process.extractOne(course_name, courses)[2]
        distances, indices = model.kneighbors(combined_matrix[idx], n_neighbors=len(courses))
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

def hit_rate(train, test, i_data, model, k=10):
    # The names of users who have taken more than one course
    usernames = train['username'].value_counts()
    usernames = usernames[usernames > 1].index

    hits = []
    for name in usernames:
        predictions = predict(name, i_data, train, model, k)['Course'].tolist()
        results = test[test['username'] == name]['course']
        test_predictions = [ result in predictions for result in results]
        isHit = [ True if pred == True else False for pred in test_predictions][0]
        hits.append(isHit)

    hits = np.count_nonzero(hits)
    accuracy = hits / len(usernames)
    
    return accuracy

def f1_score(train, test, i_data, model, k=10):
    # The names of users who have taken more than one course
    usernames = train['username'].value_counts()
    usernames = usernames[usernames > 1].index

    hits_of_users = []
    for name in usernames:
        predictions = predict(name, i_data, train, model, k)['Course'].tolist()
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