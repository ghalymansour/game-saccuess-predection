from MileStone1_Helper_Functions import *
from sklearn.model_selection import train_test_split
from Show_Correlations import show_correlations_before_pre_processing,\
    show_correlations_after_pre_processing
from datetime import datetime
import pandas as pd
# first We Read Data From Csv File
data = pd.read_csv("predicting_mobile_game_success_train_set.csv")

# After Finishing the Pre_processing we Will Show the New Correlations
# We Make Shallow Copy of the data Frame
shallow_copy_before = data.copy(deep=False)
show_correlations_before_pre_processing(shallow_copy_before)


# Make List With Columns We need Remove
Not_Needed_Features = ["URL", "ID", "Name", "Subtitle", "Icon URL", "Description", "Unnamed: 18"]

# This indicates the Start Time of the pre_processing
start_pre_process = datetime.now()
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


print("The Pre_Process Took : ", datetime.now() - start_pre_process)
# After That Make the Label The Last Column in The data Frame  of Pandas
# using the list Comprehension
data = data[[c for c in data if c not in ["Average User Rating"]] + ["Average User Rating"]]

# After Finishing the Pre_processing we Will Show the New Correlations
shallow_copy_after = data.copy(deep=False)
show_correlations_after_pre_processing(shallow_copy_after)

# we Finished All Pre_Processing On Data Let's Get The Features In X and Label In Y
# take All Columns Except the Last One
X = np.array(data.iloc[:, :-1])
# Take the Label We Want To Predict
Y = np.array(data["Average User Rating"])

# Make the Features X and Label Y as 2D array to train Model
X = X.reshape(len(data), len(data.columns) - 1)
Y = np.expand_dims(Y, axis=1)


# Normalize Our Data To make All in Specific Range
X = standard_norm(X, number_of_genres)

# All Data Are Ready Now To Train Model
# Split Data To Training data and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# multi Feature Linear regression
multi_linear_regression(x_train, x_test, y_train, y_test)
print("----------------------------------------------------------------")

# Normal Equation Regression
normal_equation_regression(X, Y)
print("---------------------------------------------------")

# Poly regression
poly_regression(x_train, x_test, y_train, y_test)
