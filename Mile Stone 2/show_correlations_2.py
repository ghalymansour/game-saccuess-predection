from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# This Function To Show The correlations of the data before pre_processing
def show_correlations_before_pre_processing(data):
    # Make List Of Columns To Be Encoded
    col = ("Languages", "Primary Genre", "Genres",
           "Current Version Release Date",
           "Original Release Date", "Age Rating", "Developer",
           "Description", "Icon URL", "Name", "URL", "Subtitle",
           "In-app Purchases")

    # Drop Empty Cells To Encode Strings Data
    data.dropna(axis=0, how="any", inplace=True)

    # loop on each column in the data set
    for i in col:
        # Encode Each Column (As There is A lot Of Strings)
        le = LabelEncoder()
        le.fit(data[i])
        data[i] = le.transform(data[i])

    # Get Correlations
    corr = data.corr()
    # Set the Size of the figure
    plt.subplots(figsize=(30, 30))
    # Make the Heat Map (Figure of Correlations)
    sns.heatmap(corr, annot=True)
    # Show the Figure
    plt.show()


# This Function To Show Correlations After Pre_Processing
def show_correlations_after_pre_processing(data):
    # Get Correlations
    corr = data.corr()
    # Set the Size of the figure
    plt.subplots(figsize=(30, 30))
    # Make the Heat Map (Figure of Correlations)
    sns.heatmap(corr, annot=True)
    # Show the Figure
    plt.show()
