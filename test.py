import sys
sys.path.append('src/isne-recommendation')

import Hybrid
import pandas as pd

# Test the function
i_data = pd.read_excel('uploads/Coursera.xlsx')
ui_data = pd.read_excel('uploads/dataset.xlsx')
recommendations = Hybrid.get_recommendations('Martha Long',i_data, ui_data, 10)
print(recommendations)