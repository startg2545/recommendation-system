from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

def fit(i_data):
    courses = i_data['course']
    courses = courses.sort_values()
    indexes = courses.index
    description = i_data['description'].iloc[indexes]
    description = description.set_axis(range(len(description)))
    model = TfidfVectorizer().fit_transform(description)
    return model

def predict(username, i_data, ui_data, model, top_n):

    def get_courses():
        courses = i_data['course']
        courses = courses.sort_values()
        courses = courses.set_axis(range(len(courses)))
        return courses

    def get_all_recommended_courses(selected_courses):
        courses = get_courses()

        # Get all indices for the target dataframe column and drop any duplicates
        indices = courses[courses.isin(selected_courses)].index

        # Get the cosine similarity scores of all courses
        cosine_similarities = linear_kernel(model)
        
        # Use boolean indexing to select rows corresponding to selected courses
        mask = np.isin(range(len(courses)), indices)

        sim_scores = cosine_similarities[mask, :]
        
        # Get the top similar courses for each selected course
        recommended_courses = []
        for idx, course in enumerate(selected_courses):
            scores = sim_scores[idx]
            recommendations = pd.DataFrame({
                'Course': courses,
                'Score': scores
            })
            recommended_courses.append(recommendations)

        return recommended_courses
    
    user_course = pd.DataFrame({
        'User': pd.Series(ui_data['username']),
        'Course': pd.Series(ui_data['course']),
    })

    selected_user_name = user_course.loc[user_course['User'] == username]
    selected_courses = selected_user_name['Course']
    
    recommended_courses = get_all_recommended_courses(selected_courses)
    
    if len(recommended_courses) == 0:
        return 'No recommendations available for this user.'

    # Prepare the final recommendations
    final_df = pd.DataFrame({
        'Course': [],
        'Score': [],
    }).rename_axis('Index')

    for x in recommended_courses:
        final_df = final_df._append(x)

    # Sort and drop duplicated courses
    final_df = final_df.sort_values('Score', ascending=False).drop_duplicates('Course')

    # Drop courses that a user has already had
    for course in selected_courses:
        final_df = final_df[final_df['Course'] != course]

    # Drop courses that a score is zero
    final_df = final_df[final_df['Score'] > 0]
    
    return final_df.head(top_n)

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
        ceil_value = np.floor( ( 1 - test_size ) * row['course_count'] - 1 )
        if course_counter <= ceil_value:
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
    
    usernames = train['username'].drop_duplicates()

    hits = []
    for name in usernames:
        predictions = predict(name, i_data, train, model, k)
        if type(predictions) is str:
            isHit = False
            hits.append(isHit)
        else:
            predictions = predictions['Course'].tolist()
            results = test[test['username'] == name]['course']
            test_predictions = [ result in predictions for result in results]
            isHit = [True if pred == True else False for pred in test_predictions]
            if True in isHit:
                hits.append(True)

    accuracy = len(hits) / len(usernames)
    
    return accuracy

def f1_score(train, test, i_data, model, k=10):
    # The names of users who have taken more than one course
    usernames = train['username'].value_counts()
    usernames = usernames[usernames > 1].index

    hits_of_users = []
    for name in usernames:
        predictions = predict(name, i_data, train, model, k)
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