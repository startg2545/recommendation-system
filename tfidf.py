#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

# In[2]:


user_input = sys.argv[1]


# Get ui_dataset and i_dataset from pickle

# In[3]:


with open('./pickle/ui_dataset.pickle', 'rb') as f:
    ui_dataset = pickle.load(f)


# In[4]:


ui_dataset_file_name = './uploads/' + ui_dataset
my_user_item = pd.read_excel(ui_dataset_file_name)


# In[5]:


with open('./pickle/i_dataset.pickle', 'rb') as f:
    i_dataset = pickle.load(f)


# In[6]:


i_dataset_file_name = './uploads/' + i_dataset
my_item = pd.read_excel(i_dataset_file_name)


# Language convert function

# In[3]:

# In[7]:


def translate_eng(text):
    return ts.translate_text(text)


# In[8]:


def is_english(text):
    for char in text:
        if char.isalpha() and char.isascii():
            return True
    return False            


# Clean title with regular expresion

# In[4]:

# In[9]:


def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# Create dataframe of courses

# In[5]:

# Take the series of courses from dataset column

# In[59]:

selected_courses = my_user_item['course'].drop_duplicates()
content = my_item['course']
content = content[content.isin(selected_courses)].drop_duplicates()
courses = content.sort_values()
course_indexes = courses.index
courses = courses.set_axis(range(0,len(content)))


# Take the description of all courses

# In[83]:


descriptions = my_item['description'].iloc[course_indexes]
descriptions = descriptions.set_axis(range(0,len(descriptions)))


# # Check if the course is in Thai language or not<br>
# is_english_courses = courses.apply(is_english)<br>
# thai_courses_not_trans = courses[is_english_courses == False]

# # Translate courses in a thai language to en english language<br>
# thai_courses = thai_courses_not_trans.apply(translate_eng)<br>
# english_courses = courses[is_english_courses == True]

# # Combine 2 series into a single series<br>
# combined_courses = thai_courses._append(english_courses)

# # Convert combined courses to be in form of regular expression<br>
# courses_clean = combined_courses.apply(clean_title).sort_index()

# In[6]:

# In[84]:


number_of_courses = len(courses)


# Create tfidf matrix

# In[7]:

# In[85]:


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(descriptions)


# dense_tfidf_matrix = tfidf_matrix.toarray()<br>
# print(dense_tfidf_matrix)

# feature_names = vectorizer.get_feature_names_out()<br>
# print(feature_names)

# Create path

# In[8]:

# In[86]:


import os
# Specify the folder path
folder_path = '/workspaces/recommendation-system/pickle'


# Create the folder if it doesn't exist

# In[87]:


if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# Export variable

# In[9]:

# In[88]:


file_path = os.path.join(folder_path, 'tfidf_matrix.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(tfidf_matrix, f)


# Create cosine similarities

# In[10]:

# In[89]:


cosine_similarities = linear_kernel(tfidf_matrix)


# Create recommendation system function

# In[11]:

# In[90]:


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

# In[92]:


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

# In[93]:


def recommender_tfidf_by_user(user_name):
    
    n_recommendations = 10
    
    df = {
        'User': pd.Series(my_user_item['username']),
        'Course': pd.Series(my_user_item['course'])
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

    # Sort courses an drop duplicated courses
    df =  df.sort_values('Score', ascending=False).drop_duplicates('Course')

    # Drop courses that user already has
    for course in selected_courses:
        df = df[df['Course'] != course]

    # Drop courses that the score value is zero
    df = df[df['Score'] != 0]
    return df.head(n_recommendations)


# In[15]:

# In[94]:


print(recommender_tfidf_by_user(user_input).to_html(index=False))


# References

# https://practicaldatascience.co.uk/data-science/how-to-create-content-recommendations-using-tf-idf<br>
# https://lukkiddd.com/tf-idf-%E0%B8%84%E0%B8%B3%E0%B9%84%E0%B8%AB%E0%B8%99%E0%B8%AA%E0%B8%B3%E0%B8%84%E0%B8%B1%E0%B8%8D%E0%B8%99%E0%B8%B0-dd1e1568312e

# Training Test Part

# Define how much similarity is to be recommended

# In[16]:

# In[ ]:


threshold_value = 0.5  # Assume the similarity is symmetric


# Calculate X and y

# In[17]:

# In[ ]:


X = tfidf_matrix.toarray()


# Create a DataFrame from courses and item_list

# In[ ]:


def get_mean(item):
    arr = recommender_tfidf(item)['Cosine Similarity Score']
    if arr.empty:
        return 0
    else:
        return arr.mean()
    


# In[ ]:


def get_label(item):
    mean = get_mean(item)
    if mean > threshold_value:
        return 'recommended'
    else:
        return 'not recommended'
    
y = [ get_label(course) for course in courses ]


# Export variable

# In[18]:

# In[ ]:


file_path = os.path.join(folder_path, 'tfidf_X.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(X, f)


# In[ ]:


file_path = os.path.join(folder_path, 'tfidf_y.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(y, f)


# Reference

# https://www.datacamp.com/tutorial/naive-bayes-scikit-learn<br>
# https://chat.openai.com/share/a3144868-3e0d-4584-b443-b6c49efb9117
