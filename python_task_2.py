#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[16]:


df= pd.read_csv('dataset-3.csv')


# In[17]:


df.head(10)


# In[18]:


df.isna().sum()


# ## Question 1: Distance Matrix Calculation

# In[38]:


import pandas as pd

def calculate_distance_matrix(df):
    # Create a pivot table to represent distances between IDs
    distance_matrix = pd.pivot_table(df, values='distance', index='id_start', columns='id_end', fill_value=0)

    # Ensure the matrix is symmetric
    distance_matrix = distance_matrix + distance_matrix.transpose()

    # Set only diagonal values to 0
    for i in distance_matrix.index:
        distance_matrix.at[i, i] = 0

    # Replace NaN values with 0 for proper sum calculation
    distance_matrix = distance_matrix.fillna(0)

    # Calculate cumulative distances along known routes
    distance_matrix = distance_matrix.apply(lambda col: col.cumsum(), axis=1)

    # Set the diagonal values to 0
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0

    return distance_matrix

# Assuming df_distance is your DataFrame with the distance information
df_distance = pd.DataFrame({
    'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418],
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420],
    'distance': [9.7, 20.2, 16.0, 21.7, 11.1, 15.6, 18.2, 13.2, 13.6, 12.9]
})

distance_matrix = calculate_distance_matrix(df_distance)
print(distance_matrix)


# ## Question 2: Unroll Distance Matrix

# In[54]:




def unroll_distance_matrix(distance_matrix):
    # Get the upper triangle of the distance matrix excluding the diagonal
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))

    # Stack the upper triangle to convert it into a long format
    unrolled_distance = upper_triangle.stack().reset_index()

    # Rename the columns
    unrolled_distance.columns = ['id_start', 'id_end', 'distance']

    return unrolled_distance

result_matrix_unrolled = unroll_distance_matrix(result_matrix)
print(result_matrix_unrolled)


# ## Question 3: Finding IDs within Percentage Threshold

# In[55]:



reference_value = 1001402  # Replace with the desired reference value
result_ids, average_distance = find_ids_within_ten_percentage_threshold(df, reference_value)

print("Result IDs within 10% threshold:", result_ids)
print("Average distance for the reference value:", average_distance)


# ## Question 4: Calculate Toll Rate

# In[56]:


def calculate_toll_rate(df):
    # Rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        column_name = f'{vehicle_type}_toll_rate'
        df[column_name] = df['distance'] * rate

    return df


df_with_toll_rates = calculate_toll_rate(df)


print(df_with_toll_rates)


# ## Question 5: Calculate Time-Based Toll Rates

# In[59]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors
    time_ranges_weekdays = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]

    time_ranges_weekends = [
        (time(0, 0, 0), time(23, 59, 59), 0.7)
    ]

    # Create columns for start_day, start_time, end_day, end_time
    df['start_day'] = np.nan
    df['start_time'] = pd.to_datetime(df['start_day'])
    df['end_day'] = np.nan
    df['end_time'] = pd.to_datetime(df['end_day'])

    # Iterate through each unique (id_start, id_end) pair
    unique_pairs = df[['id_start', 'id_end']].drop_duplicates()
    for _, row in unique_pairs.iterrows():
        id_start, id_end = row['id_start'], row['id_end']

        # Create a full 24-hour period and span all 7 days of the week
        for day_offset in range(7):
            current_date = datetime(2023, 1, 1) + timedelta(days=day_offset)

            for start_time, end_time, discount_factor in (time_ranges_weekdays if current_date.weekday() < 5 else time_ranges_weekends):
                start_datetime = datetime.combine(current_date, start_time)
                end_datetime = datetime.combine(current_date, end_time)

                # Update the DataFrame with start_day, start_time, end_day, and end_time
                mask = (df['id_start'] == id_start) & (df['id_end'] == id_end) & (df['start_time'] >= start_datetime) & (df['end_time'] <= end_datetime)
                df.loc[mask, 'start_day'] = current_date.strftime('%A')
                df.loc[mask, 'start_time'] = start_datetime
                df.loc[mask, 'end_day'] = current_date.strftime('%A')
                df.loc[mask, 'end_time'] = end_datetime

                # Update the vehicle columns based on the discount factor
                df.loc[mask, ['moto_toll_rate', 'car_toll_rate', 'rv_toll_rate', 'bus_toll_rate', 'truck_toll_rate']] *= discount_factor

    return df


df_with_time_based_toll_rates = calculate_time_based_toll_rates(df_with_toll_rates)

# Display the updated DataFrame
print(df_with_time_based_toll_rates)


# In[ ]:




