import pandas as pd
from MileStone1_Helper_Functions import *
import pickle as pk
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
# First Read Data to pre_process

data = pd.read_csv("samples_mobile_game_success_test_set.csv")

# Pre_process All Data Using Helper_functions
# Make List With Columns We need Remove
Not_Needed_Features = ["URL", "ID", "Name", "Subtitle", "Icon URL", "Description"]

# we need ro drop those Columns
# call Function drop_columns
data = drop_columns(data, Not_Needed_Features)

# Pre_processing Section

# pre_process the Size by replacing empty cells with median of the sizes of the games
data["Size"].fillna(data["Size"].median(), inplace=True)

# pre_process in-app purchases (The Stores Sells This Game)
data = pre_process_in_app_purchases(data)


# Now We Need To pre_process Each Features We will use
# First Pre_process Date (original and current version release date)
data = pre_process_date(data)

# Now We Need To Pre_Process The Average Rating Of User and Rating Count By Fill Empty Cells With zero as
# This Means no one Played This Game Or Has Any Rating Un till Now

# We Pre_Process the Age Rating By Removing The + from the Column and there is No Missing Values
data = pre_process_age_rating(data)

# Pre_Process The Developers By Encoding Names Of Each Developer(Not Unique May One Develop More Than one Game)
# any Empty Cells Of Developer we Will Drop It
data = pre_process_developer(data)

# pre_process Average User Rating, count and Price
data = pre_process_count_rating_price(data)

# pre_process the Language Of Each Game in the Data Frame
data = pre_process_languages(data)

# Each Game Has Languages So We Will Count The Number of Languages Of Each Game
data, number_of_genres = pre_process_genres(data)


data = data[[c for c in data if c not in ["Average User Rating"]] + ["Average User Rating"]]

# Take The X_Test And Y_Test From Data
x_test = np.array(data.iloc[:, :-1])
y_test = np.array(data.iloc[:, -1])

# Make Each one as 2D Arrays
x_test = x_test.reshape(len(data), len(data.columns) - 1)
y_test = np.expand_dims(y_test, axis=1)

# Normalize X
x_test = standard_norm(x_test, number_of_genres)

# Now Load Each Model To Test
linear_model = pk.load(open("C:/Users/Prince Of Persia/Desktop/Machine Template/models_regression/"
                            "Multi_Linear_Regression.sav", "rb"))
poly_model = pk.load(open("C:/Users/Prince Of Persia/Desktop/Machine Template/models_regression/"
                          "Poly_Regression_2.sav", "rb"))

# Get Prediction of Each Model
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_test)
prediction_1 = linear_model.predict(x_test)
prediction_2 = poly_model.predict(x_poly)

# Print The MSE Of Each Model
print("Multi Feature MES is : ", metrics.mean_squared_error(prediction_1, y_test))
print("Poly MSE is : ", metrics.mean_squared_error(prediction_2, y_test))
