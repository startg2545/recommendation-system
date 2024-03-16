from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

def get_recommendations(username, i_data, ui_data, top_n):

    def get_courses():
        courses = i_data['course']
        courses = courses.sort_values()
        courses = courses.set_axis(range(len(courses)))
        return courses

    def fit_transform():
        courses = i_data['course']
        courses = courses.sort_values()
        indexes = courses.index
        description = i_data['description'].iloc[indexes]
        description = description.set_axis(range(len(description)))
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