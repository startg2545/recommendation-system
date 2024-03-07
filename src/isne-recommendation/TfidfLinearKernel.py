from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

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

    
    user_course = pd.DataFrame({
        'User': pd.Series(ui_data['username']),
        'Course': pd.Series(ui_data['course']),
    })

    selected_user_name = user_course.loc[user_course['User'] == username]
    selected_courses = selected_user_name['Course']

    def get_all_recommended_courses(course):
        # Return indices for the target dataframe column and drop any duplicates
        courses = get_courses()
        indices = pd.Series(courses).drop_duplicates()
        
        # Get the index of the course
        count = 0
        for i in indices:
            if i == course:
                break
            else:
                count += 1
        target_index = count

        # Get the cosine similarity scores of the course
        cosine_similarities = fit_transform()
        sim_scores = list(enumerate(cosine_similarities[target_index]))

        # Sort the courses based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the tuple of the closest scores excluding the target item and index
        sim_scores = sim_scores[1:len(courses)]

        # Extract the tuple course_names
        index = (x[0] for x in sim_scores)
        scores = (x[1] for x in sim_scores)

        # Extract the tuple course_names
        course_indices = [i[0] for i in sim_scores]

        # Get the recommendations
        recommendations = courses.iloc[course_indices]

        # Remain specific columns
        recommendations = pd.DataFrame(tuple(zip(index, recommendations, scores)),
                                       columns=['Index', 'Course', 'Score'])
        
        # Take index from column 'index'
        idx = recommendations['Index']

        # Set and sort index
        recommendations = recommendations.set_index(idx).drop(columns=['Index']).sort_index()

        return recommendations
    
    recommended_courses = [ get_all_recommended_courses(x) for x in selected_courses ]

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