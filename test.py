import sys
sys.path.append('src/isne-recommendation')

import FeatureRatingsKNN as knn
import pandas as pd

# Test the function
i_data = pd.read_excel('uploads/Coursera.xlsx')
ui_data = pd.read_excel('uploads/dataset.xlsx')
recommendations = knn.get_recommendations('Martha Long', ui_data, 10)
print(recommendations)