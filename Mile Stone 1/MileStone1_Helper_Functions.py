from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime
import pickle as pk


# This function is used to drop not needed columns from pandas data frame
def drop_columns(data, not_needed_list):
    data.drop(not_needed_list, axis=1, inplace=True)
    return data


# This function is used to pre_process date
def pre_process_date(data):
    idx_current = data.columns.get_loc("Current Version Release Date")
    idx_original = data.columns.get_loc("Original Release Date")

    # We Can Not fill any Samples That Are Empty with Date
    data.dropna(axis=0, how="any", subset=["Original Release Date"], inplace=True)
    # Fill the Empty Of The Current Date with -1 And Then Fill Them with the original date
    count_row = 0
    data["Current Version Release Date"].fillna("-1", inplace=True)
    for i in data["Current Version Release Date"]:
        if i == "-1":
            data.iat[count_row, idx_current] = data.iat[count_row, idx_original]
        count_row += 1

    # First Make New Column of our data frame with the duration of the game in day/month/year
    # This Column is called Duration
    # Initially With the values in current version release
    data["Duration"] = data["Current Version Release Date"]
    # get index of those columns

    idx_duration = data.columns.get_loc("Duration")

    # loop on Each Row Of those Two Column  (current , original)
    for i in range(len(data["Current Version Release Date"])):
        # get the data of each row in two column
        original = data.iat[i, idx_original]
        current = data.iat[i, idx_current]

        # split each data as days / months / years
        original_list = original.split('/')
        current_list = current.split('/')

        # get  the number of years to add them in the duration column
        year_current_number = int(current_list[2])
        year_original_number = int(original_list[2])
        # get he number of Days to add them in the duration column
        day_current_number = int(current_list[0])
        day_original_number = int(original_list[0])

        # get he number of months to add them in the duration column
        month_current_number = int(current_list[1])
        month_original_number = int(original_list[1])

        # call this function to calculate the exact duration
        day, month, year = calculate_duration(year_current_number, year_original_number, month_current_number,
                                              month_original_number, day_current_number, day_original_number)

        # add the information to the new column at the cell (using i and idx)
        data.iat[i, idx_duration] = str(day) + ":" + str(month) + ":" + str(year)

    # we Don't Need The Two Columns Any More So We Will Drop Them
    data.drop(["Current Version Release Date", "Original Release Date"], axis=1, inplace=True)

    # Encode Data In Column Of Duration
    le = LabelEncoder()
    le.fit(data["Duration"])
    data["Duration"] = le.transform(data["Duration"])
    return data


# This Function To Calculate Exactly The duration of Each Game in Months, Days, Years
def calculate_duration(year_c, year_o, month_c, month_o, day_c, day_o):
    # make List of days in each month
    months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # calculate the number of days
    if day_c < day_o:
        month_c -= 1
        day_c += months_days[month_c]
    day = day_c - day_o
    # calculate the number of months
    if month_c < month_o:
        year_c -= 1
        month_c += 12
    month = month_c - month_o
    # calculate the number of years
    year = year_c - year_o
    return day, month, year


# This Function To Pre_Process the Genres By Making  Column Of Each One
def pre_process_genres(data):
    # First We Drop Samples That Are Empty in Both Primary Genre and Genres
    data.dropna(axis=0, how="all", subset=["Genres", "Primary Genre"])
    # fill the empty cells in only the Genres column with -1 to indicate it
    data["Genres"].fillna("-1", inplace=True)
    # loop on the -1 Cells in Genres and fill it with list of the value in primary Genre
    idx_primary = data.columns.get_loc("Primary Genre")
    idx_genre = data.columns.get_loc("Genres")
    count_row = 0
    for i in data["Genres"]:
        if i == "-1":
            # make list with the value in the primary genre
            list_of_item = [data.iat[count_row, idx_primary]]
            data.iat[count_row, idx_genre] = list_of_item
        count_row += 1
    # Second Step : Get All Unique Genres From This Column
    # Empty List to store Unique Genres
    unique_genres = []
    # loop on each sample
    for i in data["Genres"]:
        # split each Row To its Languages in List
        row_genres = i.split(",")
        # loop on each language in this sample
        for g in row_genres:
            # If This Language is not in our list then append it
            if g not in unique_genres:
                unique_genres.append(g)
    # Make Columns With The Genres in Our Data Frame with initially (0) value
    for i in range(len(unique_genres)):
        data[unique_genres[i]] = 0

    # After We made Those Columns In the Data Set we Need To make Each Sample If the Genre is Found To 1
    # To Count Each Row of Our Sample
    row_count = 0
    for i in data["Genres"]:
        # Split Each Sample To List of Genres
        row_genre = i.split(",")
        # Loop On Each Genre Of Our Genre Columns
        for j in unique_genres:
            # If We Found It We Change the Value from 0 to 1
            if j in row_genre:
                # Get the index of the Selected Genre
                idx = data.columns.get_loc(j)
                data.iat[row_count, idx] = 1
        row_count += 1
    # After This We Don't Need The Column Language Any More So Drop it and drop the primary genre column

    data.drop(["Genres", "Primary Genre"], axis=1, inplace=True)
    return data, len(unique_genres)


# This Function is used For Pre_Processing the Age Rating by Removing the + from the Number
def pre_process_age_rating(data):
    # We Will Assume That Any Game Has No Age Any One Can Play it So we will replace Empty cells with 4+
    data["Age Rating"].fillna("4+", inplace=True)
    # get the index of the Column
    idx_age_rating = data.columns.get_loc("Age Rating")
    # To Count Each Row in the Data Frame
    row_count = 0
    for i in data["Age Rating"]:
        # Take the Number By Splitting it from the + sign
        number_str = i.split("+")
        # Casting the number
        age = int(number_str[0])
        # Put the number in its cell
        data.iat[row_count, idx_age_rating] = age
        row_count += 1
    return data


# This function is used to pre_process the Languages Column by Placing Each Cell by the Number of Languages
# As the very Few Number of Empty Rows in this column we drop any empty sample in  this column
def pre_process_languages(data):
    # Any Empty Cell will Be Replaced By 0 Languages
    data["Languages"].fillna("-1", inplace=True)
    # To Count Each Row in the Data Sample
    count_row = 0
    # Get index of Languages Column
    idx_lang = data.columns.get_loc("Languages")
    for i in data["Languages"]:
        # Tis mean This Cell Was Empty Cell
        if i == "-1":
            # replace it with 0
            data.iat[count_row, idx_lang] = 0
        else:
            # Split Languages to List
            row_languages = i.split(",")
            # Get the number of Languages and place It in the Cell
            data.iat[count_row, idx_lang] = len(row_languages)
        count_row += 1
    return data


# This Function is used  to pre_process  Columns of Average user Rating, and price
#  Fill Empty Cells With mean
def pre_process_count_rating_price(data):
    # Fill Price Column
    data["Price"].fillna(data["Price"].median(), inplace=True)
    # Fill Average Rating Column with median (As The Label Will Not Contain Out Layers)
    data["Average User Rating"].fillna(data["Average User Rating"].mean(), inplace=True)
    # Fill Average Rating Count
    data["User Rating Count"].fillna(data["User Rating Count"].median(), inplace=True)
    return data


# This Function  to pre_process Developer Column By Encoding Each Name
def pre_process_developer(data):
    # fill Any Sample Contains Empty Cell in Developer with Unknown
    data["Developer"].fillna("UnKnown", inplace=True)

    # Encode Developer Names
    le = LabelEncoder()
    le.fit(data["Developer"])
    data["Developer"] = le.transform(data["Developer"])
    return data


# This Function To Pre_process In-App Purchases Column
def pre_process_in_app_purchases(data):
    # first Fill the empty cells in This column  with -1 to indicate it
    data["In-app Purchases"].fillna("-1", inplace=True)

    # Get index of the column
    idx_app = data.columns.get_loc("In-app Purchases")

    # Row_count to count Each Row in the Loop
    row_count = 0
    # loop on the column to pre_process it
    for i in data["In-app Purchases"]:
        # If It's Empty cell So We Will Put it with 0 (The Game Just for Test Un till Now )
        if i == "-1":
            data.iat[row_count, idx_app] = 0
        # the Game Has Stores
        else:
            # Make List of Numbers (The List Length Indicate How Many Stores Sells This Game)
            stores_list = i.split(",")
            data.iat[row_count, idx_app] = len(stores_list)
        row_count += 1
    return data


# We Will Apply Z-score  Normalization
def standard_norm(x, number_of_genres):
    # first 8 Features(Languages, price, user_count, developer, duration, size, age_rating, in_app,)
    x_features = np.empty([x.shape[0], 8])
    for i in range(8):
        # d dof to be zero mean and 1 variance
        x_features[:, i] = stats.zscore(x[:, i], axis=0, ddof=1)
    # Apply PCA To Save The Features Values and Reduce the Number of Features for both languages and genres
    # the number of Genres(Number of columns we want to apply PCA on it)
    n = 0
    if number_of_genres % 2 == 0:
        n = 2
    else:
        n = number_of_genres
    pca_genres = PCA(n_components=n)
    # reduce the number of genres to only 8 columns
    temp_genres = pca_genres.fit_transform(x[:, 8:])
    # Concatenate The New Columns To Our Features
    x_features = np.c_[x_features, temp_genres]
    return x_features


# Multi_Feature Regression
def multi_linear_regression(x_train, x_test, y_train, y_test):
    print("Multi Feature Linear Regression Data : ")
    cls = linear_model.LinearRegression()
    # Get The Start Time of the Training
    start_train_time = datetime.now()
    '''''''''
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
     '''''''''
    cls.fit(x_train, y_train)

    # After Fitting The Data Save Model To the File
    # pk.dump(cls, open("Multi_Linear_Regression.sav", "wb"))

    print("The Total Time for training is : ", datetime.now() - start_train_time)
    # Get the start Time of the Testing
    start_test_time = datetime.now()
    prediction = cls.predict(x_test)
    print("The Total Time for testing  is : ", datetime.now() - start_test_time)
    print("The MSE is : ", metrics.mean_squared_error(y_test, prediction))
    print("The Intercept is : ", cls.intercept_)
    print("The Coefficeints are : ", cls.coef_)
    print("The R2 Score is : ", metrics.r2_score(y_test, prediction))


# Ploy Regression
def poly_regression(x_train, x_test, y_train, y_test):
    n = int(input("Please Enter The Degree Of the Polynomial Regression : "))
    poly_features = PolynomialFeatures(degree=n)

    print("Poly Regression Data : ")
    cls = linear_model.LinearRegression()
    # Get The Start Time of the Training
    start_train_time = datetime.now()
    '''''''''
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    '''''''''
    cls.fit(poly_features.fit_transform(x_train), y_train)
    # After Fitting The Data Save Model To the File
    # pk.dump(cls, open("Poly_Regression.sav", "wb"))

    print("The Total Time for training is : ", datetime.now() - start_train_time)
    # Get the start Time of the Testing
    start_test_time = datetime.now()
    prediction = cls.predict(poly_features.fit_transform(x_test))
    print("The Total Time for testing  is : ", datetime.now() - start_test_time)
    print("The MSE is : ", metrics.mean_squared_error(y_test, prediction))
    print("The Intercept is : ", cls.intercept_)
    print("The Coefficeints are : ", cls.coef_)
    print("The R2 Score is : ", metrics.r2_score(y_test, prediction))


def normal_equation_regression(x, y):
    # Before it concatenate with X the the X0 (Node) All With ones
    x = np.c_[np.ones([x.shape[0], 1]), x]

    # Split X and y to train and Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    # get Weights
    start_train_time = datetime.now()
    '''''''''
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    '''''''''
    x_trans = np.transpose(x_train)
    theta = np.dot(np.linalg.pinv(np.dot(x_trans, x_train)), np.dot(x_trans, y_train))

    # After Fitting The Data Save Model To the File
    # pk.dump(theta, open("Normal_Equation.sav", "wb"))
    print("The Total Time for training is : ", datetime.now() - start_train_time)
    # Get the start Time of the Testing
    start_test_time = datetime.now()
    # Get prediction  H_theta(X)
    prediction = np.dot(x_test, theta)
    print("The Total Testing Time is : ", datetime.now() - start_test_time)
    print("Normal Equation Regression Data : ")
    print("The MSE is : ", metrics.mean_squared_error(y_test, prediction))
    print("The Intercept is : ", theta[0])
    print("The Coefficeints are : ", theta)
    print("The R2 Score is : ", metrics.r2_score(y_test, prediction))
