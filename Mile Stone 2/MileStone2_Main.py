import MileStone2_Helper_Functions as Mh
import pandas as pd
from datetime import datetime
from show_correlations_2 import show_correlations_before_pre_processing, show_correlations_after_pre_processing
# first Read file CSV OF Data set
data = pd.read_csv("appstore_games_classification.csv")

# After Finishing the Pre_processing we Will Show the New Correlations
# We Make Shallow Copy of the data Frame
shallow_copy_before = data.copy(deep=False)
show_correlations_before_pre_processing(shallow_copy_before)

# This Indicate the Start Time Of Pre_processing
start_process = datetime.now()
# Make List of Un needed columns
Not_Needed_Features = ["URL", "ID", "Name", "Subtitle", "Icon URL", "Description",
                       "Primary Genre"]

# we need ro drop those Columns
# call Function drop_columns
data = Mh.drop_columns(data, Not_Needed_Features)


# Pre_processing Section

# pre_process in-app purchases (The Stores Sells This Game)
data = Mh.pre_process_in_app_purchases(data)

# # Each Game Has Languages So We Will Count The Number of Languages Of Each Game
data = Mh.pre_process_languages(data)

# Now We Need To pre_process Each Features We will use
# First Pre_process Date (original and current version release date)
data = Mh.pre_process_date(data)

# Pre_Process Genres By Making Each Genre With Column in the Data Frame of Pandas
data = Mh.pre_process_genres(data)

# We Pre_Process the Age Rating By Removing The + from the Column and there is No Missing Values
data = Mh.pre_process_age_rating(data)

# Pre_Process The Developers By Encoding Names Of Each Developer(Not Unique May One Develop More Than one Game)
# any Empty Cells Of Developer we Will Drop It
data = Mh.pre_process_developer(data)

# pre_process Average User Rating, count and Price
data = Mh.pre_process_count_rating_price(data)

# pre_process The Classes Of data Frame Make Each class With number -> High(2), Low(0), Intermediate(1)
data = Mh.pre_process_classes(data)

# After That Make the Label The Last Column in The data Frame  of Pandas
# using the list Comprehension
data = data[[c for c in data if c not in ["Rate"]] + ["Rate"]]

# After Finishing the Pre_processing we Will Show the New Correlations
shallow_copy_after = data.copy(deep=False)
show_correlations_after_pre_processing(shallow_copy_after)


# This Indicate The Total Amount Of Time The Pre_Process Took
print("The Pre_processing Took : ", datetime.now() - start_process)

# get The Label Of Our data Set To Help Us Get Each Class In Data Frame
Y = data.iloc[:, -1]

# we Get The Data Of Each Class In Pandas Data Frame of (High, Low, Intermediate)
High_data = data.loc[Y == 2]  # High Class
Low_data = data.loc[Y == 0]  # Low Class
Intermediate_data = data.loc[Y == 1]  # Intermediate Class

# After We get The Three Classes And pre_processing All data We Will Call One Vs One Classification
Mh.one_vs_one_classification(High_data, Intermediate_data, Low_data)

# Then Classify Between The Three Classes Using One vs All Classification
Mh.one_vs_all_classification(High_data, Intermediate_data, Low_data)

