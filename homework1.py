import streamlit as st               

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - Pedicting Housing Prices Using Regression")

#############################################

st.markdown("Welcome to the Practical Applications in Machine Learning (PAML) Course! You will build a series of end-to-end ML pipelines, working with various data types and formats, and will need to engineer your system to support training, testing, and deploying ML models.")

st.markdown("""The goal of this assignment is to build a regression machine learning pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Note, that the ‘Explore Dataset’ and ‘Preprocess Data’ steps from homework 1 can be re-used for this assignment.

The learning outcomes for this assignment are:
- Build end-to-end regression pipeline using 1) multiple regression, 2) polynomial regression, and 3) ridge regression.
- Evaluate regression methods using standard metrics including root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).
- Develop a web application that walks users through steps of the regression pipeline and provide tools to analyze multiple methods across multiple metrics. 
""")

st.markdown(""" California Housing Data

This assignment involves testing the end-to-end pipeline in a web application using a California Housing dataset from the textbook: Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media, Inc., 2022 [GitHub,Dataset Description]. The dataset was captured from California census data in 1990. We have added additional features to the dataset. The features include:
- longitude: longitudinal coordinate
- latitude: latitudinal coordinate
- housing_median_age: median age of district
- total_rooms: total number of rooms per district
- total_bedrooms: total number of bedrooms per district
- population: total population of district
- households: total number of households per district'
- median_income: median income
- ocean_proximity: distance from the ocean
- median_house_value: median house value
- city: city location of house
- county: county of house
- road: road of the house
- postcode: zip code 

""")

st.markdown("Click **Explore Dataset** to get started.")