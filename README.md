# examen_nivel_intermedio_Henry_Larreal
# Data Science Tasks

This repository contains a collection of Python functions and examples for various data science tasks. The tasks cover data manipulation, regression modeling, and utility functions commonly used in data analysis and machine learning projects.

## Table of Contents

1. [Task 1: DataFrame Filtering](#task-1-dataframe-filtering)
2. [Task 2: Generate Regression Data](#task-2-generate-regression-data)
3. [Task 3: Train Multiple Linear Regression Model](#task-3-train-multiple-linear-regression-model)
4. [Task 4: Flatten Nested List](#task-4-flatten-nested-list)
5. [Task 5: Group and Aggregate DataFrame](#task-5-group-and-aggregate-dataframe)
6. [Task 6: Train Logistic Regression Model](#task-6-train-logistic-regression-model)
7. [Task 7: Apply Function to DataFrame Column](#task-7-apply-function-to-dataframe-column)
8. [Task 8: Filter and Square List](#task-8-filter-and-square-list)

## Task 1: DataFrame Filtering

The `filtered_df` function filters a DataFrame based on a specified column and threshold value.

```python
def filtered_df(df, clm, Umbreon):
    filtered_df = df[df[clm] > Umbreon]
    return filtered_df
```

## Task 2: Generate Regression Data

The `generate_regression_data` function creates synthetic data for regression analysis, including both independent and dependent variables.

```python
def generate_regression_data(nsamp):
    # Function implementation...
    return df_indi, df_dep
```

## Task 3: Train Multiple Linear Regression Model

This task involves training a multiple linear regression model using scikit-learn and saving it to a file.

```python
def train_multiple_linear_regression(x, y):
    # Function implementation...
    return multiple_linear_model
```

## Task 4: Flatten Nested List

The `flatten_list` function uses list comprehension to flatten a nested list.

```python
def flatten_list(nested_number_l):
    return [item for sublist in nested_number_l for item in sublist]
```

## Task 5: Group and Aggregate DataFrame

The `group_and_aggregate` function performs grouping and aggregation operations on a DataFrame.

```python
def group_and_aggregate(df, gp_cl, ac_cl):
    result = df.groupby(gp_cl)[ac_cl].mean()
    return result
```

## Task 6: Train Logistic Regression Model

This task involves training a logistic regression model and saving it to a file.

```python
def train_logistic_regression(x, y):
    model_t = LogisticRegression()
    model_t.fit(x, y)
    return model_t
```

## Task 7: Apply Function to DataFrame Column

The `apply_function_to_column` function applies a custom function to a specified column in a DataFrame.

```python
def apply_function_to_column(df, cl_name, func):
    df[cl_name] = df[cl_name].apply(func)
    return df
```

## Task 8: Filter and Square List

The `filter_and_square` function filters a list for values greater than 5 and squares them.

```python
def filter_and_square(num):
    return [x**2 for x in num if x > 5]
```

## Requirements

- pandas
- scikit-learn
- faker

## Usage

Each function can be imported and used in your Python scripts. Some functions include example usage in comments within the code.

## Note

This README provides an overview of the functions available in the repository. For detailed usage and implementation, please refer to the individual function docstrings and comments in the source code.
