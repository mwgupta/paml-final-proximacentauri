[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/1PiV8uhi)
# Practical Applications in Machine Learning 

# Homework 1: Predict Housing Prices Using Regression

The goal of this assignment is to build a regression machine learning pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Note, that the ‘Explore Dataset’ and ‘Preprocess Data’ steps from homework 0 can be re-used for this assignment.

* Evaluate regression methods using standard metrics including root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).
* Plot learning curves to inspect the cost function, detect underfitting, overfitting, to identify an ideal model.

# Assignment Deadline:
* Due: February 12, 2025 at 11:00PM 
* What to turn in: Submit responses on GitHub AutoGrader
* Assignment Type: Group (Up to 5 members)
* Total Points: 12 points

## Assignment Outline
* California Housing Dataset
* Train Regression Models (9 points) 
* Model Evaluation (3 points) 
* Test Code with Github Autograder

# Reading Prerequisite 

* Read Chapter 4 and review the jupyter notebook on Training Models of “Machine Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow.” O’Reilly Media, Inc., 2022. 

# Join Github Classrooom

* Create a team name for Github Classroom
* Clone your groups repository when joining the Github Classroom for Homework 1

# End-to-End Regression Models for Housing Prices 

# California Housing Data

This assignment involves testing the end-to-end pipeline in a web application using a California Housing dataset from the textbook: Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media, Inc., 2022. The dataset was captured from California census data in 1990. We have added additional features to the dataset. The features include:
* longitude - longitudinal coordinate
* latitude - latitudinal coordinate
* housing_median_age - median age of district
* total_rooms - total number of rooms per district
* total_bedrooms - total number of bedrooms per district
* population - total population of district
* households - total number of households per district'
* median_income - median income
* ocean_proximity - distance from the ocean
* median_house_value - median house value
* city - city location of house
* county - county of house
* road - road of the house
* postcode - zip code 

The Github repository contains two datasets in the dataset directory:
* housing_paml - Use the California Housing dataset for testing HW1.

# Training Models (12 points)

Complete the checkpoint functions for the following regression classes in the pages folder:
* LinearRegression class
* PolynomialRegression class
* RidgeRegression class
* LassoRegression class

# Testing Code with Github Autograder

Test your homework solution as needed using Github Autograder. Clone your personal private copy of the homework assignment. Once the code is downloaded and stored on your computer in your directory of choice, you can start testing the assignment. To test your code, open a terminal and navigate to the directory of the homework1.py file. Then, type ‘pytest’ ad press enter. The autograder with print feedback to inform you what checkpoint functions are failing. Test homework1.py using this command in your terminal.

To run all test:

```
pytest
```

To run in verbose mode:

```
pytest -v
```

To run an specific checkpoint (i.e., checkpoint1):

```
pytest -m checkpoint1
```

To start and visualize your assignment's progress a in web interface:
```
streamlit run homework1.py
```

# Reflection Assessment

Complete on Canvas.

# Further Issues and questions ❓

If you have issues or questions, don't hesitate to contact the teaching team:

* Angelique Taylor, Instructor, amt298@cornell.edu
* Jonathan Segal, Teaching Assistant, jis62@cornell.edu 
* Marianne Arriola, Teaching Assistant, ma2238@cornell.edu
* Adnan Al Armouti, Teaching Assistant, aa2546@cornell.edu
* Jacky He, Grader, ph474@cornell.edu
* Yibei Li, Grader, yl3692@cornell.edu
* Stella Hong, Grader, sh2577@cornell.edu
