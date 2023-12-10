#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[81]:


df= pd.read_csv('dataset-1.csv')


# In[82]:


df.head(10)


# ## Question 1: Car Matrix Generation

# In[83]:


df1 = pd.DataFrame(df)
df1.head()


# In[84]:


df1_pivot = pd.pivot(df1,'id_1','id_2','car').fillna(0)


# In[85]:


df1_pivot


# In[86]:


filtered_df = df1_pivot[df1_pivot < 15]
filtered_df.count().count()


# ## Question 2: Car Type Count Calculation

# In[87]:



def get_type_count(df):
    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each car_type category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count

# Assuming df1 is your DataFrame from 'dataset-1.csv'
# Replace df1 with your actual DataFrame
result = get_type_count(df1)
print(result)


# ## Question 3: Bus Count Index Retrieval

# In[88]:



def get_bus_indexes():
    # Read the dataset into a DataFrame
    df = pd.read_csv('dataset-1.csv')

    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where the 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the list of indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage:
result_bus_indexes = get_bus_indexes()
print(result_bus_indexes)


# ## Question 4: Route Filtering

# In[89]:




def filter_routes(df):
    # Group by 'route' and calculate the mean of the 'truck' column
    route_means = df.groupby('route')['truck'].mean()
    
    # Filter routes where the mean of 'truck' column is greater than 7
    selected_routes = route_means[route_means > 7]
    
    # Return the sorted list of selected routes
    return selected_routes.index.sort_values().tolist()

# Assuming df is your DataFrame from dataset-1.csv
selected_routes = filter_routes(df)
print(selected_routes)


# ## Question 5: Matrix Value Modification

# In[90]:


import pandas as pd

# Assuming df1 is your DataFrame from 'dataset-1.csv'

# Function to generate car matrix
def generate_car_matrix(df):
    car_matrix = pd.pivot_table(df, values='car', index='id_1', columns='id_2', fill_value=0)
    car_matrix.values[[range(car_matrix.shape[0])]*2] = 0
    return car_matrix

# Function to multiply matrix values
def multiply_matrix(car_matrix):
    modified_matrix = car_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)
    return modified_matrix

# Generate car matrix
car_matrix_result = generate_car_matrix(df)

# Multiply matrix values and round
modified_matrix_result = multiply_matrix(car_matrix_result)

print(modified_matrix_result.head())


# In[116]:


res1 = pd.DataFrame(modified_matrix_result)
res1.head(10)


# In[91]:


sf= pd.read_csv('dataset-2.csv')


# In[92]:


sf


# In[95]:


sf.head(10)


# In[96]:


sf.isna().sum()


# In[98]:


sf.fillna(0,inplace = True)


# In[100]:


sf.isna().sum()


# In[105]:


sf.head(20)


# In[111]:


import pandas as pd

def clean_and_analyze_time_data(df):
    # Convert 'startDay' and 'endDay' to datetime format
    df['startDay'] = pd.to_datetime(df['startDay'], errors='coerce')
    df['endDay'] = pd.to_datetime(df['endDay'], errors='coerce')

    # Correct timestamp values to follow the "hh:mm:ss" format
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M:%S')
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M:%S')

    # Perform time check analysis
    df['start_timestamp'] = df['startDay'] + pd.to_timedelta(df['startTime'])
    df['end_timestamp'] = df['endDay'] + pd.to_timedelta(df['endTime'])

    # Check if each (id, id_2) pair has incorrect timestamps
    result_series = (df['start_timestamp'].dt.time != pd.to_datetime('00:00:00').time()) |                     (df['end_timestamp'].dt.time != pd.to_datetime('23:59:59').time())

    # Create a DataFrame with multi-index
    result_df = pd.DataFrame({'is_complete': ~result_series, 'day_coverage': df['startDay'].dt.day_name()})
    result_df.index = pd.MultiIndex.from_frame(df[['id', 'id_2']])

    return result_df

# Assuming sf is your DataFrame from 'dataset-2.csv'
result_df = clean_and_analyze_time_data(sf)
print(result_df)


# In[115]:


res2 = pd.DataFrame(result_df)
res2.head(10)


# In[ ]:




