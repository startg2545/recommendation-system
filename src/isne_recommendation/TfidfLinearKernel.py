from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

def get_recommendations(username, i_data, ui_data, top_n):

    def get_courses():
        selected_courses = ui_data['course'].drop_duplicates()
        content = i_data['course']
        content = content[content.isin(selected_courses)].drop_duplicates()
        courses = content.sort_values()
        courses = courses.set_axis(range(0, len(courses)))
        return courses

    def fit_transform():
        courses = get_courses()
        indexes = courses.index
        description = i_data['description'].iloc[indexes]
        description = description.set_axis(range(0, len(description)))
        matrix = TfidfVectorizer().fit_transform(description)
        cosine_similarities = linear_kernel(matrix)
        return cosine_similarities

    def get_all_recommended_courses(selected_courses):
        courses = get_courses()

        # Get all indices for the target dataframe column and drop any duplicates
        indices = courses[courses.isin(selected_courses)].index

        # Get the cosine similarity scores of all courses
        cosine_similarities = fit_transform()
        
        # Use boolean indexing to select rows corresponding to selected courses
        mask = np.isin(range(len(courses)), indices)
        sim_scores = cosine_similarities[mask, :]
        
        # Get the top similar courses for each selected course
        recommended_courses = []
        for idx, course in enumerate(selected_courses):
            target_index = courses[courses == course].index[0]
            
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

def get_evaluations(df):
    # Separate features (X) and target variable (y)
    X = df.drop(columns=['course'])
    y = df['course']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the text data in X_train and X_test using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train['description'])
    X_test_transformed = vectorizer.transform(X_test['description'])

    # Initialize and train a classifier (e.g., Logistic Regression)
    classifier = LogisticRegression()
    classifier.fit(X_train_transformed, y_train)

    # Make predictions on the testing data
    y_pred = classifier.predict(X_test_transformed)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy