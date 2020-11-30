from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from datetime import datetime


# This function is used to drop not needed columns from pandas data frame
def drop_columns(data, not_needed_list):
    data.drop(not_needed_list, axis=1, inplace=True)
    return data


# This function is used to pre_process date
def pre_process_date(data):
    # We Can Not fill any Samples That Are Empty with Date
    data.dropna(axis=0, how="any", subset=["Current Version Release Date",
                                           "Original Release Date"], inplace=True)

    # First Make New Column of our data frame with the duration of the game in day/month/year
    # This Column is called Duration
    # Initially With the values in current version release
    data["Duration"] = data["Current Version Release Date"]
    # get index of those columns
    idx_current = data.columns.get_loc("Current Version Release Date")
    idx_original = data.columns.get_loc("Original Release Date")
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
    # First We Drop Any Samples we can not fill it
    data.dropna(axis=0, how="any", subset=["Genres"], inplace=True)

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
    # After This We Don't Need The Column Language Any More So Drop it
    data.drop("Genres", axis=1, inplace=True)
    return data


# This Function is used For Pre_Processing the Age Rating by Removing the + from the Number
def pre_process_age_rating(data):
    # We Can not Fill Data Of Age Rating With Any Empty Samples (Cells) with proper Value so we will drop it
    data.dropna(axis=0, how="any", subset=["Age Rating"], inplace=True)
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
    # First Drop Samples In Data Frame For Games Not Have Languages (Can Not Fill It With Proper Value, and very few )
    data.dropna(axis=0, subset=["Languages"], inplace=True)
    # To Count Each Row in the Data Sample
    count_row = 0
    # Get index of Languages Column
    idx_lang = data.columns.get_loc("Languages")
    for i in data["Languages"]:
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
    # We Can Not fill the Label with Any Proper Value So we Will Drop the Empty cells Of Label
    data.dropna(axis=0, subset=["Rate"], inplace=True)
    data["Rate"].fillna("Intermediate", inplace=True)
    # Fill Average Rating Count
    data["User Rating Count"].fillna(data["User Rating Count"].median(), inplace=True)
    return data


# This Function  to pre_process Developer Column By Encoding Each Name
def pre_process_developer(data):
    # fill Any Sample Contains Empty Cell in Developer with Unknown
    data["Developer"].fillna("UnKnown", inplace=True)
    # Encode the Developer Column
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


# Pre_process The Classes To Numbers
# High -> 2 , Low -> 0, Intermediate -> 1
def pre_process_classes(data):
    # Get Index Of the Rate
    idx_class = data.columns.get_loc("Rate")
    # To Count Each Row where we stand
    count_row = 0
    for i in data["Rate"]:
        class_name = str(i)
        if class_name == "High":
            data.iat[count_row, idx_class] = 2
        elif class_name == "Low":
            data.iat[count_row, idx_class] = 0
        elif class_name == "Intermediate":
            data.iat[count_row, idx_class] = 1
        count_row += 1
    return data


# We Will Apply Z-score  Normalization
def standard_norm(x):
    x_features = np.empty([x.shape[0], x.shape[1]])
    for i in range(7):
        # d dof to be zero mean and 1 variance
        x_features[:, i] = stats.zscore(x[:, i], axis=0, ddof=1)
    return x_features


# This Function To Get The Train and Test Samples from Each Class
def get_train_test_data(data):
    # Get Features of The data
    x = np.array(data.iloc[:, :-1])
    # Get Label Of Data and make it as type int to cast the object classes
    y = np.array(data.iloc[:, -1].astype(int))
    # Reshape The Size of X as Samples (Rows) and Features(Columns)
    x = x.reshape(len(data), len(data.columns) - 1)
    # Make New Dimension for Y Labels
    y = np.expand_dims(y, axis=1)
    # Normalize Data Using Z Score
    x = standard_norm(x)
    # Split To Test And train Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


# This function is used to call All Types We used Of Classification in (one vs one approach)
def one_vs_one_classification(high_class, intermediate_class, low_class):
    # Using Logistic Regression  :
    print("One Vs One Logistic Regression : ")
    # First Call logistic one vs one function to classify between high and low class
    print("Logistic Regression one_vs_one (High and Low Classes) : ")
    one_vs_one_logistic(high_class, low_class)
    # Second Call logistic one vs one function to classify between high and InterMediate class
    print("Logistic Regression one_vs_one (High and InterMediate) : ")
    one_vs_one_logistic(high_class, intermediate_class)
    # Third Call logistic one vs one function to classify between Low and InterMediate class
    print("Logistic Regression one_vs_one (Intermediate and Low) : ")
    one_vs_one_logistic(low_class, intermediate_class)
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # Using SVM :
    print("SVM one_vs_one : ")
    one_vs_one_svm(high_class, intermediate_class, low_class)
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # Using KNN :
    print("KNN on_vs_one : ")
    print("KNN One_Vs_One with (High and Low) Classes : ")
    one_vs_one_knn(high_class, low_class)
    print("KNN One_Vs_One with (High and intermediate) Classes : ")
    one_vs_one_knn(high_class, intermediate_class)
    print("KNN One_Vs_One with (Intermediate and Low) Classes : ")
    one_vs_one_knn(intermediate_class, low_class)
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # Using AdaBoost With Tree :
    print("Adaboost with decision Tree  One_vs_one : ")
    print("AdaBoost Using (High and Intermediate) Classes : ")
    one_vs_one_adb(high_class, intermediate_class)
    print("AdaBoost Using (High and Low) Classes : ")
    one_vs_one_adb(high_class, low_class)
    print("AdaBoost Using (Low and Intermediate) Classes : ")
    one_vs_one_adb(low_class, intermediate_class)


# This function is used to call All Types We Used Of Classification in (one vs all approach)
# in this Approach  Make the Two Classes With the Same Label And one Class With Its Label To apply (one vs All)
def one_vs_all_classification(high_class, intermediate_class, low_class):
    print("****************************************************************************")
    print("****************************************************************************")
    print("****************************************************************************")
    # Using Logistic Regression  :
    print("One Vs ALL Logistic Regression : ")
    # First Classify between high and Rest
    print("Logistic Regression one_vs_all (High and Rest Classes) : ")
    one_vs_all_logistic(high_class, low_class, intermediate_class)
    # Second Classify between Low and Rest
    print("Logistic Regression one_vs_all (Low and Rest Classes) : ")
    one_vs_all_logistic(low_class, intermediate_class, high_class)
    # Third Classify between InterMediate and Rest
    print("Logistic Regression one_vs_all (InterMediate and Rest Classes) : ")
    one_vs_all_logistic(intermediate_class, low_class, high_class)
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # Using SVM :
    print("SVM one_vs_all : ")
    one_vs_all_svm(high_class, intermediate_class, low_class)
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # Using Knn :
    print("KNN one_vs_all : ")
    one_vs_all_knn(high_class, intermediate_class, low_class)

    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # Using AdaBoost With Tree :
    print("Adaboost with decision Tree  One_vs_All : ")
    one_vs_all_adb(high_class, intermediate_class, low_class)


# This Function is using Logistic Regression To classify Between Two Classes
def one_vs_one_logistic(class1, class2):
    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)
    # Make the Concatenation of Train data (X and Y)
    x_train = np.concatenate((x_1_train, x_2_train), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train), axis=0)
    # Make the Concatenation of Test Data (X, Y)
    x_test = np.concatenate((x_1_test, x_2_test), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test), axis=0)

    start_train = datetime.now()
    # Now We Can Use The Built in Logistic Regression
    lg = linear_model.LogisticRegression(solver='lbfgs')
    # fit The Model Using Training Data
    y_train = y_train.flatten()
    lg.fit(x_train, y_train)
    print("The Training Took : ", datetime.now() - start_train)
    # Test Model using Testing Data
    start_test = datetime.now()
    prediction = lg.predict(x_test).astype(int)
    # Get the Accuracy of The Model With Those Two Classes
    # Make the Two Arrays one Dimension to use the Mean function
    print("The Testing Took : ", datetime.now() - start_test)
    prediction = prediction.flatten()
    y_test = y_test.flatten()
    acc = np.mean(prediction == y_test) * 100
    print("Accuracy is : ", acc)


# This Function is using Logistic Regression To classify Between Three Classes
# Make Class 2 and Class 3 As the Same Class (Only One Class )and classify between Them and Class 1
def one_vs_all_logistic(class1, class2, class3):
    # Make Copy Of Each Class To Work on It
    class_1_copy = class1.copy(deep=False)
    class_2_copy = class2.copy(deep=False)
    class_3_copy = class3.copy(deep=False)
    # first Make The Label Of Class2 And Class 3 as the Same Label But Not (1, 0 or 2) for Example With 3 Label
    # Get the index of the label
    class_2_copy["Rate"] = class_2_copy["Rate"] = 3
    # Now Merge Two Classes
    # Ignore the index in the original classes
    class_merge = pd.concat([class_2_copy, class_3_copy], ignore_index=True)

    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class_1_copy)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class_merge)
    # Make the Concatenation of Train data (X and Y)
    x_train = np.concatenate((x_1_train, x_2_train), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train), axis=0)
    # Make the Concatenation of Test Data (X, Y)
    x_test = np.concatenate((x_1_test, x_2_test), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test), axis=0)

    # Now We Can Use The Built in Logistic Regression
    lg = linear_model.LogisticRegression(solver='lbfgs', multi_class="auto")
    # fit The Model Using Training Data
    y_train = y_train.flatten()
    start_train = datetime.now()
    lg.fit(x_train, y_train)
    print("The Training Took : ", datetime.now() - start_train)
    # Test Model using Testing Data
    start_test = datetime.now()
    prediction = lg.predict(x_test).astype(int)
    print("The Testing Took : ", datetime.now() - start_test)
    # Get the Accuracy of The Model With Those Two Classes
    # Make the Two Arrays one Dimension to use the Mean function
    prediction = prediction.flatten()
    y_test = y_test.flatten()
    acc = np.mean(prediction == y_test) * 100
    print("Accuracy is : ", acc)


# This Function for Training SVM model in one_vs_one Approach
def one_vs_one_svm(class1, class2, class3):
    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)
    # Get train and test Data Of Class 2
    x_3_train, x_3_test, y_3_train, y_3_test = get_train_test_data(class3)

    # Concatenate The Data of train and test
    x_train = np.concatenate((x_1_train, x_2_train, x_3_train), axis=0)
    x_test = np.concatenate((x_1_test, x_2_test, x_3_test), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train, y_3_train), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test, y_3_test), axis=0)

    # Flatten() Y_Train to be 1D array
    y_train = y_train.flatten()
    # Now Train Model Using One vs One Approach

    start_train = datetime.now()
    svm_model_linear_ovo = SVC(kernel='rbf', C=1, gamma=2.5).fit(x_train, y_train)
    print("The Training Took : ", datetime.now() - start_train)

    # model accuracy for X_test
    start_test = datetime.now()
    accuracy = svm_model_linear_ovo.score(x_test, y_test) * 100
    print("The Testing Took : ", datetime.now() - start_test)
    print('One VS One SVM accuracy: ' + str(accuracy))


# This Function for Training SVM model in one_vs_all Approach
def one_vs_all_svm(class1, class2, class3):
    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)
    # Get train and test Data Of Class 2
    x_3_train, x_3_test, y_3_train, y_3_test = get_train_test_data(class3)

    # Concatenate The Data of train and test
    x_train = np.concatenate((x_1_train, x_2_train, x_3_train), axis=0)
    x_test = np.concatenate((x_1_test, x_2_test, x_3_test), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train, y_3_train), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test, y_3_test), axis=0)

    # Flatten() Y_Train to be 1D array
    y_train = y_train.flatten()
    start_train = datetime.now()
    svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf', C=1, gamma=2.5)).fit(x_train, y_train)
    print("The Training Took : ", datetime.now() - start_train)
    # svm_predictions = svm_model_linear_ovr.predict(X_test)

    # model accuracy for X_test
    start_test = datetime.now()
    accuracy = svm_model_linear_ovr.score(x_test, y_test) * 100
    print("The Testing Took : ", datetime.now() - start_test)
    print('One VS Rest SVM accuracy: ' + str(accuracy))


# This function to calculate the accuracy of knn in one_vs_one Approach
def one_vs_one_knn(class1, class2):
    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)
    # Concatenate The Data of train and test
    x_train = np.concatenate((x_1_train, x_2_train), axis=0)
    x_test = np.concatenate((x_1_test, x_2_test), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test), axis=0)
    # There is Three Classes
    # We Found That The Best Hyper Parameter Value is 3
    knn = KNeighborsClassifier(n_neighbors=3)
    # Train Model Using x_train and y_train
    y_train_m = y_train.flatten()
    start_train = datetime.now()
    knn.fit(x_train, y_train_m)
    print("The Training Took : ", datetime.now() - start_train)
    # Get The Prediction
    start_test = datetime.now()
    prediction = knn.predict(x_test)
    print("The Testing Took : ", datetime.now() - start_test)
    # Get The Accuracy  by using 1D array of prediction and y_test
    y_test_m = y_test.flatten()
    prediction_m = prediction.flatten()
    accuracy = np.mean(prediction_m == y_test_m) * 100
    print("The Accuracy is : ", accuracy)


# This function to calculate the accuracy of knn in one_vs_all Approach
def one_vs_all_knn(class1, class2, class3):
    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)
    # Get train and test Data Of Class 2
    x_3_train, x_3_test, y_3_train, y_3_test = get_train_test_data(class3)

    # Concatenate The Data of train and test
    x_train = np.concatenate((x_1_train, x_2_train, x_3_train), axis=0)
    x_test = np.concatenate((x_1_test, x_2_test, x_3_test), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train, y_3_train), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test, y_3_test), axis=0)
    # There is Three Classes
    # We Found That The Best Hyper Parameter Value is 3
    knn = KNeighborsClassifier(n_neighbors=3)
    # Train Model Using x_train and y_train
    y_train_m = y_train.flatten()
    start_train = datetime.now()
    knn.fit(x_train, y_train_m)
    print("The Training Took : ", datetime.now() - start_train)
    # Get The Prediction
    start_test = datetime.now()
    prediction = knn.predict(x_test)
    print("The Testing Took : ", datetime.now() - start_test)
    # Get The Accuracy  by using 1D array of prediction and y_test
    y_test_m = y_test.flatten()
    prediction_m = prediction.flatten()
    accuracy = np.mean(prediction_m == y_test_m) * 100
    print("The Accuracy is : ", accuracy)


# This Function for Adb  In One_VS_One Approach
def one_vs_one_adb(class1, class2):

    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)

    # Concatenate The Data of train and test
    x_train = np.concatenate((x_1_train, x_2_train), axis=0)
    x_test = np.concatenate((x_1_test, x_2_test), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test), axis=0)

    # Make the Adb With the Decision tree With depth
    bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion="entropy", max_depth=40),
                             n_estimators=100)
    # Make the Y_ 1D array first
    y_train = y_train.flatten()
    start_train = datetime.now()

    bdt.fit(x_train, y_train)
    print("The Training Took : ", datetime.now() - start_train)
    start_test = datetime.now()
    y_prediction = bdt.predict(x_test)
    print("The Testing Took : ", datetime.now() - start_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The Accuracy is : " + str(accuracy))


# This Function for Adb  In One_VS_All Approach
def one_vs_all_adb(class1, class2, class3):
    # Get train and test Data of class 1
    x_1_train, x_1_test, y_1_train, y_1_test = get_train_test_data(class1)
    # Get Train and Test Data Of Class 2
    x_2_train, x_2_test, y_2_train, y_2_test = get_train_test_data(class2)
    # Get train and test Data Of Class 2
    x_3_train, x_3_test, y_3_train, y_3_test = get_train_test_data(class3)

    # Concatenate The Data of train and test
    x_train = np.concatenate((x_1_train, x_2_train, x_3_train), axis=0)
    x_test = np.concatenate((x_1_test, x_2_test, x_3_test), axis=0)
    y_train = np.concatenate((y_1_train, y_2_train, y_3_train), axis=0)
    y_test = np.concatenate((y_1_test, y_2_test, y_3_test), axis=0)

    # Make the Adb With the Decision tree With depth
    bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion="entropy", max_depth=40),
                             n_estimators=100)
    # Make the Y_ 1D array first
    y_train = y_train.flatten()
    start_train = datetime.now()
    bdt.fit(x_train, y_train)
    print("The Training Took : ", datetime.now() - start_train)
    start_test = datetime.now()

    y_prediction = bdt.predict(x_test)
    print("The Testing Took : ", datetime.now() - start_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The Accuracy is : " + str(accuracy))
