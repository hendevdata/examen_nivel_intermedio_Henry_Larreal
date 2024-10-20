# Task 1
## Filter the DataFrame based on whether the values in the specified column are greater than the threshold == UMBREON 
#Importing Necessary Libraries
import pandas as pd

def filtered_df(df, clm, Umbreon):
    #Filtering the DataFrame
    filtered_df = df[df[clm] > Umbreon]
    return filtered_df

# Task 2
## Importing the libraries --> Pandas, Faker & Random

import pandas as pd
from faker import Faker
import random

def generate_regression_data(nsamp):
    fake = Faker()

    # Independent variables
    indi_vars = {
        'A': [],
        'B': [],
        'C': []
    }

    # Dependent variables
    dep_vars = []

    # Generate independent variables and dependent variable
    for _ in range(nsamp):
        A = random.uniform(0, 100)
        B = random.uniform(0, 100)
        C = random.uniform(0, 100)

        # Simulating dependent variable
        dep_value = 7 * A + 8 * B + C + random.uniform(-7, 7)

        # Adding to lists
        indi_vars['A'].append(A)
        indi_vars['B'].append(B)
        indi_vars['C'].append(C)
        dep_vars.append(dep_value)

    # Create DataFrames
    df_dep = pd.DataFrame(dep_vars, columns=["Target"])
    df_indi = pd.DataFrame(indi_vars)

    return df_indi, df_dep

# Example usage
df_indi, df_dep = generate_regression_data(10)

# Print the independent and dependent DataFrames
print("Independent Variables DataFrame:")
print(df_indi)
print("\nDependent Variables DataFrame:")
print(df_dep)


# Task 3 --> Train a multiple linear regression model with the independent and dependant data from the TASK 2 ---> 

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import time

def train_multiple_linear_regression(x, y):

    #Multiple Linear Regression Model
    multiple_linear_model = LinearRegression()

    #Train the model using indi_var and dep_var
    multiple_linear_model.fit(x, y)

    # --> Return the trained model
    return multiple_linear_model

# The DFs are already defined on the task Number 2
x = df_indi
y = df_dep ['Target'] #Series Variables

# Training the model
multiple_linear_model = train_multiple_linear_regression(x , y)

# Show the coef & inter of the model
print(multiple_linear_model.coef_)
print(multiple_linear_model.intercept_)    

# Generate file
multiple_linear_model =  'Multiple_Linear_Regression_model.pkl'
with open(multiple_linear_model, 'wb') as file:
    pickle.dump(multiple_linear_model, file)

# Task 4 --> Create a function called flatten list what use Comprehension list
# to flatten a list of lists

def flatten_list(nested_number_l):
    return [item for sublist in nested_number_l for item in sublist]

#Use and Test
nested_number_l = [[1, 2, 3], [4, 5, 6], [23,24,28], [51,57,32],[500,300,20495,25,9123]]
print(flatten_list(nested_number_l))

# Task 5 --> Creating  the function group and aggregate .mean()

import pandas as pd 

def group_and_aggregate(df, gp_cl, ac_cl):
    result =  df.groupby(gp_cl)[ac_cl].mean()
    return result
'''
#Test --> please ignore!
data = {
    'gp_cl': ['Caracas', 'Maracaibo', 'Trujillo', 'Valencia', 'Miranda', 'Maracay'],
    'ac_cl': [5450, 120, 1200, 70, 5, 23],
    'prep':  [2300, 3200,  5000, 4500, 38000, 54]

}
df = pd.DataFrame(data)

result = group_and_aggregate(df, 'gp_cl', 'ac_cl' )

print(result)
'''

#Task -->6 Create a function with LogisticRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(x, y):
    model_t = LogisticRegression()
    model_t.fit(x,y)
    return model_t

df_indi = pd.DataFrame({
    'A': [15, 22, 36, 41, 55],
    'B': [24, 31, 42, 55, 61],
    'C': [54, 64, 71, 89, 99]
})
df_dep = pd.Series([0, 1, 0, 1, 0])

model_t = train_logistic_regression(df_indi, df_dep)
print(model_t.coef_)
print(model_t.intercept_)

model_t =  'Logistic_regression.pkl'
with open(model_t, 'wb') as file:
    pickle.dump(LogisticRegression, file)


# Task 7 

import pandas as pd 

def apply_function_to_column(df, cl_name, func):
    df[cl_name] = df[cl_name].apply(func)
    
    return df
'''
#Use and sample
data = {
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
}
df = pd.DataFrame(data)

def custom_function(x):
    return x * 2

df_m = apply_function_to_column(df, 'A', custom_function)

print(df_m)
'''

#Task 8 --> Square List

def filter_and_square(num):
    return [x**2 for x in num if x > 5]
'''
#Test
list = [1,6,4,3,1]
re = filter_and_square(list)
print(re)
'''








