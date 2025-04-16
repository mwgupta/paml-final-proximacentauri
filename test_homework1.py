from pages import B_Train_Model, C_Test_Model
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

############## Assignment 1 Inputs #########
student_filepath = "datasets/housing_dataset.csv"
grader_filepath = "test_dataframe_file/housing_dataset.csv"
student_dataframe = pd.read_csv(student_filepath)
grader_dataframe = pd.read_csv(grader_filepath)
e_dataframe = pd.read_csv(grader_filepath)
test_metrics = ['mean_absolute_error', 'root_mean_squared_error', 'r2_score']



@pytest.fixture
def mock_env_setup():
    mock_predict = Mock(return_value=1)
    mock_model = Mock(predict=mock_predict)
    with patch('D_Deploy_App.st.session_state', {'deploy_model': mock_model}) as mock_session_state:
        yield mock_session_state


@pytest.fixture
def L_model():
    model = B_Train_Model.LinearRegression(learning_rate=0.001, num_iterations=500)
    model.W = np.zeros(3)
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2],[4]])
    model.num_examples = 2
    return model
    

@pytest.fixture
def P_model():
    model = B_Train_Model.PolynomailRegression(degree=3, learning_rate=0.001, num_iterations=500)
    model.W = np.zeros(3)
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2], [4]])
    return model

@pytest.fixture
def R_model():
    model = B_Train_Model.RidgeRegression(learning_rate=0.4, num_iterations=100, l2_penalty=0.5)
    # model.W = np.zeros((3)) 
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2], [4]])
    model.num_examples = model.X.shape[0]
    model.Y_pred=None
    return model

@pytest.fixture
def LR_model():
    model = B_Train_Model.LassoRegression(learning_rate=0.4, num_iterations=100, l1_penalty=0.5)
    # model.W = np.zeros((3)) 
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2], [4]])
    model.num_examples = model.X.shape[0]
    model.num_features = model.X.shape[1]
    model.Y_pred=None
    return model

def Test_data():
    y_true = np.array([3,-0.5,2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    return y_true, y_pred
# Checkpoint 1
@pytest.mark.checkpoint1
def test_split_dataset():
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    e_Y = e_dataframe.loc[:, e_dataframe.columns.isin(['median_house_value'])]
    s_split_train_x, s_split_train_y, s_split_test_x, s_split_test_y = B_Train_Model.split_dataset(
        e_X, e_Y, 30)
    assert s_split_train_x.shape == (14448,16)

# Checkpoint 2 
@pytest.mark.checkpoint2
def test_predict_Linear(L_model):
    L_model.W = np.array([1, 8.10, 10.11, 1.13])
    expected_output = np.array([ -18.33999941,  20.33999941]).reshape(-1,1)

    predictions = L_model.predict(L_model.X)

    np.testing.assert_array_almost_equal(predictions, expected_output,
                                         decimal=6, err_msg="Error in prediction method, debug!!", verbose=True)

# Checkpoint 3 
@pytest.mark.checkpoint3
def test_update_weights_Linear(L_model):
    L_model.W = np.array([1, 8.10, 10.11, 1.13])
    old_weights = np.copy(L_model.W)
    L_model = L_model.update_weights()
    W = np.array([[ 1.004  ],
                  [ 8.06332],
                  [10.07332],
                  [ 1.09332]])
    assert not np.array_equal(old_weights, L_model.W)
    assert L_model is not None
    assert isinstance(L_model, B_Train_Model.LinearRegression)
    np.testing.assert_array_almost_equal(L_model.W, W, decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)

# checkpoint 4 
@pytest.mark.checkpoint4
def test_fit_Linear(L_model):
    np.set_printoptions(precision=15)
    updated_weights = np.array([[1.897466235428521],
                                [0.316886934767022],
                                [0.316886935739613],
                                [0.316886937520929]])
    L_model = L_model.fit(L_model.X, L_model.Y)
    np.testing.assert_array_almost_equal(L_model.W, updated_weights, decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)


# Checkpoint 5
@pytest.mark.checkpoint5
def test_get_weights(L_model):
    L_model= L_model.fit(L_model.X,L_model.Y)
    print(L_model.W)
    model_name = 'Multiple Linear Regression'
    features = ['feature1', 'feature2', 'feature3']
    
    expected_out_dict = {
        'Multiple Linear Regression': np.array([[1.897466235428521],
       [0.316886934767022],
       [0.316886935739613],
       [0.316886937520929]]),
        'Polynomial Regression': [],
        'Ridge Regression': []
    }
  
    result = L_model.get_weights(model_name, features)
    np.testing.assert_array_almost_equal(result[model_name], expected_out_dict[model_name],decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)


# Checkpoint 6 Polynomial
@pytest.mark.checkpoint6
def test_fit_Polynomial(P_model):
    np.set_printoptions(precision=10)
    #updated_weights = np.array([ 0.16736985, -0.07086507, 0.12156239, 0.25148682])
    updated_weights = np.array([[1.8974662354],
        [0.0526315776],
        [0.0526315777],
        [0.052631578 ],
        [0.0526315791],
        [0.0526315792],
        [0.0526315792],
        [0.0526315792],
        [0.0526315792],
        [0.0526315792],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793]])

    P_model = P_model.fit(P_model.X, P_model.Y)

    np.testing.assert_array_almost_equal(P_model.W, updated_weights, decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)


# Checkpoint 7
@pytest.mark.checkpoint7
def test_predict_Polynomial(P_model):
    expected_predictions = np.array([[0.897466], [ 2.897466]])
    P_model = P_model.fit(P_model.X, P_model.Y)
    predictions = P_model.predict(P_model.X)
    np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=6, err_msg="Oh no! Debug!!!,Prediction error!", verbose=True)


# Checkpoint 8
@pytest.mark.checkpoint8
def test_update_weights_Ridge(R_model):
    #np.set_printoptions(precision=15)
    R_model.W = np.array([1, 8.10, 10.11, 1.13])
    old_weights = np.copy(R_model.W)
    R_model.update_weights()
    ## code pushing done
    expected_weights = np.array([[  2.8     ],
                  [ -4.951998],
                  [ -2.539998],
                  [-13.315999]]) 
    assert not np.array_equal(old_weights, R_model.W)
    assert R_model is not None
    assert isinstance(R_model, B_Train_Model.RidgeRegression)
    np.testing.assert_array_almost_equal(R_model.W, expected_weights, decimal=6, err_msg="Oh no! Debug!!!,Update weight error!", verbose=True)


# Checkpoint 9
@pytest.mark.checkpoint9
def test_update_weights_Lasso(LR_model):
    #np.set_printoptions(precision=15)
    LR_model.W = np.array([1, 8.10, 10.11, 1.13])
    old_weights = np.copy(LR_model.W)
    LR_model.update_weights()
    ## code pushing done
    expected_weights = np.array([[  2.5     ],
                  [ -6.671998],
                  [ -4.661998],
                  [-13.641999]])    
    assert not np.array_equal(old_weights, LR_model.W)
    assert LR_model is not None
    assert isinstance(LR_model, B_Train_Model.LassoRegression)
    np.testing.assert_array_almost_equal(LR_model.W, expected_weights, decimal=6, err_msg="Oh no! Debug!!!,Update weight error!", verbose=True)

# checkpoint 10
@pytest.mark.checkpoint10
def test_rmse():
    y_true, y_pred =Test_data() 
    error = C_Test_Model.rmse(y_true, y_pred)
    expected_error = 0.6123724356957945
    assert np.isclose(error, expected_error, atol=1e-8), f"Debug!!!,Expected error: {expected_error}, but got: {error}"

# checkpoint 11   
@pytest.mark.checkpoint11 
def test_mae():
    y_true, y_pred =Test_data() 
    error = C_Test_Model.mae(y_true, y_pred)
    expected_error = 0.5
    assert np.isclose(error, expected_error, atol=1e-8), f"Debug!!!,Expected error: {expected_error}, but got: {error}"

# checkpoint 12
@pytest.mark.checkpoint12 
def test_r2():
    y_true, y_pred =Test_data() 
    error = C_Test_Model.r2(y_true, y_pred)
    expected_error = 0.9486081370449679
    assert np.isclose(error, expected_error, atol=1e-8), f"Debug!!!,Expected error: {expected_error}, but got: {error}"

# checkpoint 12
'''
@pytest.mark.checkpoint12 
def test_deploy_model(mock_env_setup):
    # Create a dummy DataFrame as input
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    predicted_price = D_Deploy_App.deploy_model(df)
    assert predicted_price == 1, "Check your session state loading carefully"
'''

# def test_transform(P_model):
#     #X = np.array([8.1, 10.1, 13.1])
#     #model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
#     #model.Y = np.array([2, 4])
#     P_model.num_examples = P_model.X.shape[0]
#     expected = np.array([[1.000000000000000e+00, 2.000000000000000e+00, 3.400000000000000e+00,
#                       4.800000000000000e+00, 4.000000000000000e+00, 6.800000000000000e+00,
#                       9.600000000000000e+00, 1.156000000000000e+01, 1.632000000000000e+01,
#                       2.304000000000000e+01, 8.000000000000000e+00, 1.360000000000000e+01,
#                       1.920000000000000e+01, 2.312000000000000e+01, 3.264000000000000e+01,
#                       4.608000000000000e+01, 3.930399999999999e+01, 5.548799999999999e+01,
#                       7.833600000000000e+01, 1.105920000000000e+02],
#                       [1.000000000000000e+00, 8.100000000000000e+00, 1.013000000000000e+01,
#                        1.310000000000000e+01, 6.561000000000000e+01, 8.205300000000000e+01,
#                        1.061100000000000e+02, 1.026169000000000e+02, 1.327030000000000e+02,
#                        1.716100000000000e+02, 5.314409999999999e+02, 6.646293000000001e+02,
#                        8.594910000000000e+02, 8.311968900000001e+02, 1.074894300000000e+03,
#                        1.390041000000000e+03, 1.039509197000000e+03, 1.344281390000000e+03,
#                        1.738409300000000e+03, 2.248091000000000e+03]])

#     # expected = np.array([[1.000000e+00, 8.100000e+00, 6.561000e+01, 5.314410e+02],
#     #                                     [1.000000e+00, 1.010000e+01, 1.020100e+02, 1.030301e+03],
#     #                                     [1.000000e+00, 1.310000e+01, 1.716100e+02, 2.248091e+03],])
    
#     X_transformed = P_model.transform(P_model.X)
#     #print(np.array(X_transformed))
#     #mismatches = np.where(expected != X_transformed)
#     #print(mismatches)
#     np.testing.assert_array_almost_equal(X_transformed, expected, err_msg="Debug!!!, transform did not provide correct answer, delve deeper.", decimal=6, verbose =True)






# # CheckPoint 7
# @pytest.mark.checkpoint7

# # Checkpoint 8
# @pytest.mark.checkpoint8
# def test_inspect_coefficient():
#     student_dataframe_nan = A_Explore_Preprocess_Dataset.remove_nans(student_dataframe)

#     X = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin([
#                                                                         'longitude'])]
#     Y = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin(
#         ['median_house_value'])]
#     X_train, X_val, y_train, y_val = B_Train_Model.split_dataset(X, Y, 30)
#     student_Multiple_Linear_Regression = B_Train_Model.train_multiple_regression(
#         X_train, y_train, 'Multiple Linear Regression')
#     student_Polynomial_Regression = B_Train_Model.train_polynomial_regression(
#         X_train, y_train, 'Polynomial Regression', 3, False)
#     ridge_params = {"solver": ["auto", "svd"], "alpha": [1, 0.5]}
#     lasso_params = {"tol": [0.001, 0.0001], "alpha": [1, 0.5]}
#     student_Ridge_Regression = B_Train_Model.train_ridge_regression(
#         X_train, y_train, 'Ridge Regression', ridge_params=ridge_params, ridge_cv_fold=3)
#     student_Lasso_Regression = B_Train_Model.train_lasso_regression(
#         X_train, y_train, 'Lasso Regression', lasso_params=lasso_params, lasso_cv_fold=3)
#     expected_poly_coefficient = np.array(
#         [27815.07044908, -19450.82482728, -19777.94476361])
#     expected_ridge_coefficient = np.array([[2569.7090167]])
#     expected_lasso_coefficient = np.array([2569.05265257])
#     regression_methods_options = ['Multiple Linear Regression',
#                                   'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
#     models = {'Multiple Linear Regression': student_Multiple_Linear_Regression,
#               'Polynomial Regression': student_Polynomial_Regression,
#               'Ridge Regression': student_Ridge_Regression,
#               'Lasso Regression': student_Lasso_Regression}

#     student_results = B_Train_Model.inspect_coefficients(
#         models, regression_methods_options
#     )

#     student_Multiple_Linear_Regression_coeff = np.array(
#         student_results['Multiple Linear Regression'])
#     student_Polynomial_Regression_coeff = np.array(
#         student_results['Polynomial Regression'])
#     student_Ridge_Regression_coeff = np.array(
#         student_results['Ridge Regression'])
#     student_Lasso_Regression_coeff = np.array(
#         student_results['Lasso Regression'])
#     assert np.allclose(student_Multiple_Linear_Regression_coeff,
#                        np.array([1369.35463082]))
#     assert np.allclose(
#         student_Polynomial_Regression[-1] .coef_, expected_poly_coefficient)
#     assert np.allclose(
#         student_Ridge_Regression['ridgeCV'].best_estimator_.coef_, expected_ridge_coefficient)
#     assert np.allclose(
#         student_Lasso_Regression['lassoCV'].best_estimator_.coef_, expected_lasso_coefficient)

# def flatten_dict(dictionary, key):
#     return np.array([val for val in dictionary[key].values()])

# # Checkpoint 9 
# @pytest.mark.checkpoint9
# def test_model_metrics():
#     expected_results = {
#         'Multiple Linear Regression': {
#             'mean_absolute_error': 84214.69579043229,
#             'root_mean_squared_error': 107405.43533055007,
#             'r2_score': 0.000707935606354182
#         },
#         'Polynomial Regression': {
#             'mean_absolute_error': 82308.29766716881,
#             'root_mean_squared_error': 105603.77140404365,
#             'r2_score': 0.03395184402618556
#         },
#         'Ridge Regression': {
#             'mean_absolute_error': 84214.68485060059,
#             'root_mean_squared_error': 107405.43533088331,
#             'r2_score': 0.0007079356001534753
#         },
#         'Lasso Regression': {
#             'mean_absolute_error': 84214.65490126661,
#             'root_mean_squared_error': 107405.43533520534,
#             'r2_score': 0.0007079355197296966
#         }
#     }

#     expected_multi = flatten_dict(
#         expected_results, 'Multiple Linear Regression')
#     expected_poly = flatten_dict(expected_results, 'Polynomial Regression')
#     expected_ridge = flatten_dict(expected_results, 'Ridge Regression')
#     expected_lasso = flatten_dict(expected_results, 'Lasso Regression')

#     student_dataframe_nan = B_Preprocess_Data.remove_nans(student_dataframe)

#     X = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin([
#                                                                         'longitude'])]
#     Y = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin(
#         ['median_house_value'])]
#     student_multi_reg = B_Train_Model.train_multiple_regression(
#         X, Y, 'Multiple Linear Regression')
#     student_poly_reg = B_Train_Model.train_polynomial_regression(
#         X, Y, 'Polynomial Regression', 3, True
#     )
#     student_ridge_reg = B_Train_Model.train_ridge_regression(
#         X, Y, 'Ridge Regression',
#         {"solver": ["auto", "svd"], "alpha": [1, 0.5]}, 3
#     )
#     student_lasso_reg = B_Train_Model.train_lasso_regression(
#         X, Y, 'Lasso Regression',
#         {"tol": [0.001, 0.0001], "alpha": [1, 0.5]}, 3
#     )

#     models = [student_multi_reg, student_poly_reg,
#               student_ridge_reg, student_lasso_reg]

#     regression_methods_options = ['Multiple Linear Regression',
#                                   'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']

#     student_results = {}
#     for idx, model in enumerate(models):
#         student_results[regression_methods_options[idx]] = D_Test_Model.compute_eval_metrics(
#             X, Y, model, test_metrics
#         )

#     print(student_results)

#     student_multi = flatten_dict(
#         student_results, 'Multiple Linear Regression')
#     student_poly = flatten_dict(student_results, 'Polynomial Regression')
#     student_ridge = flatten_dict(student_results, 'Ridge Regression')
#     student_lasso = flatten_dict(student_results, 'Lasso Regression')

#     assert np.allclose(expected_multi, student_multi)
#     assert np.allclose(expected_poly, student_poly)
#     assert np.allclose(expected_ridge, student_ridge)
#     assert np.allclose(expected_lasso, student_lasso)

# # Checkpoint 10 
# @pytest.mark.checkpoint10
# def test_plot_learning_curve():
#     student_dataframe_nan = B_Preprocess_Data.remove_nans(student_dataframe)

#     X = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin([
#                                                                         'longitude'])]
#     Y = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin(
#         ['median_house_value'])]

#     X_train, X_val, y_train, y_val = B_Train_Model.split_dataset(X, Y, 30)

#     student_Multiple_Linear_Regression = B_Train_Model.train_multiple_regression(
#         X_train, y_train, 'Multiple Linear Regression')
#     student_Polynomial_Regression = B_Train_Model.train_polynomial_regression(
#         X_train, y_train, 'Polynomial Regression', 3, False)
#     ridge_params = {"solver": ["auto", "svd"], "alpha": [1, 0.5]}
#     lasso_params = {"tol": [0.001, 0.0001], "alpha": [1, 0.5]}
#     student_Ridge_Regression = B_Train_Model.train_ridge_regression(
#         X_train, y_train, 'Ridge Regression', ridge_params=ridge_params, ridge_cv_fold=3)
#     student_Lasso_Regression = B_Train_Model.train_lasso_regression(
#         X_train, y_train, 'Lasso Regression', lasso_params=lasso_params, lasso_cv_fold=3)

#     regression_methods_options = ['Multiple Linear Regression',
#                                   'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
#     models = [student_Multiple_Linear_Regression, student_Polynomial_Regression,
#               student_Ridge_Regression, student_Lasso_Regression]

#     _, student_plot_learning_curve_mul = D_Test_Model.plot_learning_curve(
#         X_train, X_val, y_train, y_val, models[0], test_metrics, regression_methods_options[0])
#     _, student_plot_learning_curve_pol = D_Test_Model.plot_learning_curve(
#         X_train, X_val, y_train, y_val, models[1], test_metrics, regression_methods_options[1])
#     _, student_plot_learning_curve_rid = D_Test_Model.plot_learning_curve(
#         X_train, X_val, y_train, y_val, models[2], test_metrics, regression_methods_options[2])
#     _, student_plot_learning_curve_las = D_Test_Model.plot_learning_curve(
#         X_train, X_val, y_train, y_val, models[3], test_metrics, regression_methods_options[3])
#     expected_mul = pd.read_csv(
#         "./test_dataframe_file/Multiple Linear Regression Errors.csv")
#     expected_pol = pd.read_csv(
#         "./test_dataframe_file/Polynomial Regression Errors.csv")
#     expected_rid = pd.read_csv(
#         "./test_dataframe_file/Ridge Regression Errors.csv")
#     expected_las = pd.read_csv(
#         "./test_dataframe_file/Lasso Regression Errors.csv")
#     pd.testing.assert_frame_equal(
#         student_plot_learning_curve_mul, expected_mul)
#     pd.testing.assert_frame_equal(
#         student_plot_learning_curve_pol, expected_pol)
#     pd.testing.assert_frame_equal(
#         student_plot_learning_curve_rid, expected_rid)
#     pd.testing.assert_frame_equal(
#         student_plot_learning_curve_las, expected_las)