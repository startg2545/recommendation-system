#%% md
# Import required packages
#%%
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
#%% md
# Read the file
#%%
file = pd.read_excel('shortcourses2566.xlsx')
#%% md
# Clean title with regular expresion
#%%
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)
#%% md
# Create dataframe of courses
#%%
content = file['หลักสูตรอบรมระยะสั้น'].drop_duplicates().fillna('')
courses = content.sort_values().set_axis(range(0,len(content)))
courses_clean = courses.apply(clean_title)
#%% md
# Create tfidf matrix
#%%
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_clean)
#%% md
# Create cosine similarities
#%%
cosine_similarities = linear_kernel(tfidf_matrix)
#%% md
# Create recommendation system function
#%%
def recommender_tfidf(course_name, limit):
    """Return a dataframe of content recommendations based on TF-IDF cosine similarity.
    
    Args:
        courses (object): Pandas Series containing the text data. 
        column (string): Name of column used, i.e. 'title'. 
        course_name (string): Name of title to get recommendations for, i.e. 1982 Ferrari 308 GTSi For Sale by Auction
        cosine_similarities (array): Cosine similarities matrix from linear_kernel
        limit (int, optional): Optional limit on number of recommendations to return. 
        
    Returns: 
        Pandas dataframe. 
    """

    # Return indices for the target dataframe column and drop any duplicates
    indices = pd.Series(courses).drop_duplicates()
    
    # Get the index for the target course_name
    count = 0
    for name in indices:
        if name == course_name:
            break
        else:
            count = count + 1
    target_index = count

    # Get the cosine similarity scores for the target course_name
    cosine_similarity_scores = list(enumerate(cosine_similarities[target_index]))

    # Sort the cosine similarities in order of closest similarity
    cosine_similarity_scores = sorted(cosine_similarity_scores, key=lambda x: x[1], reverse=True)

    # Return tuple of the requested closest scores excluding the target item and index
    cosine_similarity_scores = cosine_similarity_scores[1:limit+1]
    
    # Extract the tuple course_names
    index = (x[0] for x in cosine_similarity_scores)
    scores = (x[1] for x in cosine_similarity_scores)

    # Get the indices for the closest items
    recommendation_indices = [i[0] for i in cosine_similarity_scores]

    # Get the actual recommendations
    recommendations = courses.iloc[recommendation_indices]

    # Return a recommendations
    recommendations = pd.DataFrame(tuple(zip(index, recommendations, scores)),
                      columns=['index','recommendation', 'cosine_similarity_score'])

    return recommendations
#%% md
# Test recommendation system using TF-IDF
#%%
recommender_tfidf('การวินิจฉัยภาวะฉุกเฉินจากอุบัติเหตุ (Diagnostic Radiology of Traumatic Emergency)', 10)
#%% md
# References
#%% md
# https://practicaldatascience.co.uk/data-science/how-to-create-content-recommendations-using-tf-idf
# https://medium.com/analytics-vidhya/how-to-do-a-content-based-filtering-using-tf-idf-f623487ed0fd