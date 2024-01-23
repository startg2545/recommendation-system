#!/usr/bin/env python
# coding: utf-8

# Import required packages

# In[1]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import translators as ts
import pickle
import sys


# Read the file

# In[2]:

# Receive the user name from app.py
user_input = sys.argv[1]

# Get filename from pickle
with open('./pickle/filename.pickle', 'rb') as f:
    filename = pickle.load(f)

file_name = './uploads/' + filename
# file_name = './uploads/' + 'Sample Dataset.xlsx'
file = pd.read_excel(file_name)


# Language convert function

# In[3]:

def translate_eng(text):
    return ts.translate_text(text)

def is_english(text):
    for char in text:
        if char.isalpha() and char.isascii():
            return True
    return False
            


# Clean title with regular expresion

# In[4]:


def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# Create dataframe of courses

# In[5]:


# Take the series of courses from dataset column
content = file['course'].drop_duplicates().fillna('')
courses = content.sort_values().set_axis(range(0,len(content)))

# Check if the course is in Thai language or not
is_english_courses = courses.apply(is_english)
thai_courses_not_trans = courses[is_english_courses == False]

# Translate courses in a thai language to en english language
thai_courses = thai_courses_not_trans.apply(translate_eng)
english_courses = courses[is_english_courses == True]

# Combine 2 series into a single series
combined_courses = thai_courses._append(english_courses)

# Convert combined courses to be in form of regular expression
courses_clean = combined_courses.apply(clean_title).sort_index()


# In[6]:


number_of_courses = len(courses)


# Create tfidf matrix

# In[7]:


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(courses_clean)

# dense_tfidf_matrix = tfidf_matrix.toarray()
# print(dense_tfidf_matrix)

# feature_names = vectorizer.get_feature_names_out()
# print(feature_names)

# Create path

# In[8]:


import os
# Specify the folder path
folder_path = '/workspaces/recommendation-system/pickle'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# Export variable

# In[9]:


file_path = os.path.join(folder_path, 'tfidf_matrix.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(tfidf_matrix, f)


# Create cosine similarities

# In[10]:


cosine_similarities = linear_kernel(tfidf_matrix)

# Create recommendation system function

# In[11]:


def recommender_tfidf(course_name):
    """Return a dataframe of content recommendations based on TF-IDF cosine similarity.
    
    Args:
        courses (object): Pandas Series containing the text data. 
        column (string): Name of column used, i.e. 'title'. 
        course_name (string): Name of title to get recommendations for, i.e. 1982 Ferrari 308 GTSi For Sale by Auction
        cosine_similarities (array): Cosine similarities matrix from linear_kernel
        
    Returns: 
        Pandas dataframe. 
    """

    # Return indices for the target dataframe column and drop any duplicates
    indices = pd.Series(courses).drop_duplicates()
    
    # Get the index for the target course_name
    target_index = indices[indices == course_name].index[0]

    # Get the cosine similarity scores for the target course_name
    cosine_similarity_scores = list(enumerate(cosine_similarities[target_index]))
    
    # Sort the cosine similarities in order of closest similarity
    cosine_similarity_scores = sorted(cosine_similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Return tuple of the requested closest scores excluding the target item and index
    cosine_similarity_scores = cosine_similarity_scores[1:number_of_courses]
    cosine_similarity_scores = [(i, score) for i, score in cosine_similarity_scores if score != 0]
    
    # Extract the tuple course_names
    index = (x[0] for x in cosine_similarity_scores)
    scores = (x[1] for x in cosine_similarity_scores)
    
    # Get the indices for the closest items
    recommendation_indices = [i[0] for i in cosine_similarity_scores]
    
    # Get the actual recommendations
    recommendations = courses.iloc[recommendation_indices]
    
    # Return a recommendations
    recommendations = pd.DataFrame(tuple(zip(index, recommendations, scores)),
                      columns=['Index','Course', 'Cosine Similarity Score'])
    
    # Take index from column 'index'
    idx = recommendations['Index']
    
    # Set and sort index
    recommendations = recommendations.set_axis(idx).drop(columns='Index')
    return recommendations


# Test recommendation system using TF-IDF

# In[12]:


# Predata for hybrid recommendation

# In[13]:


def recommender_tfidf_all_courses(course_name):
    """Return a dataframe of content recommendations based on TF-IDF cosine similarity.
    
    Args:
        courses (object): Pandas Series containing the text data. 
        column (string): Name of column used, i.e. 'title'. 
        course_name (string): Name of title to get recommendations for, i.e. 1982 Ferrari 308 GTSi For Sale by Auction
        cosine_similarities (array): Cosine similarities matrix from linear_kernel
        
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
    cosine_similarity_scores = cosine_similarity_scores[1:len(courses)]

    # Extract the tuple course_names
    index = (x[0] for x in cosine_similarity_scores)
    scores = (x[1] for x in cosine_similarity_scores)

    # Get the indices for the closest items
    recommendation_indices = [i[0] for i in cosine_similarity_scores]

    # Get the actual recommendations
    recommendations = courses.iloc[recommendation_indices]

    # Return a recommendations
    recommendations = pd.DataFrame(tuple(zip(index, recommendations, scores)),
                                   columns=['Index','Course', 'Score'])
    
    # Take index from column 'index'
    idx = recommendations['Index']
    
    # Set and sort index 
    recommendations = recommendations.set_axis(idx).drop(columns='Index').sort_index()
    return recommendations


# In[14]:


def recommender_tfidf_by_user(user_name):
    
    n_recommendations = number_of_courses - 1
    
    df = {
        'User': pd.Series(file['username']),
        'Course': pd.Series(file['course'])
    }

    user_course = pd.DataFrame(df)
    selected_user_name = user_course.loc[user_course['User'] == user_name]
    selected_courses = selected_user_name['Course']

    recommended_courses = [ recommender_tfidf_all_courses(x) for x in selected_courses]

    # pre dataframe
    df = pd.DataFrame({
        'Course': [],
        'Score': []
    }).rename_axis('Index')

    for x in recommended_courses:
        df = df._append(x)
    df =  df.sort_values('Score', ascending=False).drop_duplicates('Course')
    return df.head(n_recommendations)


# In[15]:

print(recommender_tfidf_by_user(user_input).to_html(index=False))

# References

# https://practicaldatascience.co.uk/data-science/how-to-create-content-recommendations-using-tf-idf
# https://lukkiddd.com/tf-idf-%E0%B8%84%E0%B8%B3%E0%B9%84%E0%B8%AB%E0%B8%99%E0%B8%AA%E0%B8%B3%E0%B8%84%E0%B8%B1%E0%B8%8D%E0%B8%99%E0%B8%B0-dd1e1568312e

# Training Test Part

# Define how much similarity is to be recommended

# In[16]:


threshold_value = 0.5  # Assume the similarity is symmetric


# Calculate X and y

# In[17]:


X = tfidf_matrix.toarray()

# Create a DataFrame from courses and item_list
def get_mean(item):
    arr = recommender_tfidf(item)['Cosine Similarity Score']
    if arr.empty:
        return 0
    else:
        return arr.mean()
    

def get_label(item):
    mean = get_mean(item)
    if mean > threshold_value:
        return 'recommended'
    else:
        return 'not recommended'
    
y = [ get_label(course) for course in courses ]


# Export variable

# In[18]:


file_path = os.path.join(folder_path, 'tfidf_X.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(X, f)

file_path = os.path.join(folder_path, 'tfidf_y.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(y, f)


# Reference

# https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
# https://chat.openai.com/share/a3144868-3e0d-4584-b443-b6c49efb9117
