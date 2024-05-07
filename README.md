# CS254-Project-ML
1. Our dataset is private so we are not sharing it here 
2. We have divided the code to be run step by step
3. please use the Full_Code file to reproduce the results

# Below is explanition of each function in the code: 
preprocess_text(text):
This function preprocesses a given text by converting it to lowercase, removing punctuation, numbers, and stopwords.

load_and_preprocess_data():
This function loads the raw data from CSV files, drops unnecessary columns, and selects relevant columns for further processing.

preprocess_features_and_labels(filtered_df):
This function preprocesses the features (issue descriptions) and labels (categories) extracted from the filtered dataframe.
It applies the preprocess_text() function to clean the text data, removes missing values, and filters out certain categories.

visualize_original_categories_distribution(data):
This function visualizes the distribution of original categories in the dataset using bar plots.
It calculates the count and ratio of each category and displays them in a bar plot.

train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
This function trains and evaluates different machine learning models (Linear SVM, Random Forest, Decision Tree, Naive Bayes) using TF-IDF vectorized features.
It returns the accuracy scores for each model.

train_keras_model(X_train, y_train, X_test, y_test):
This function trains and evaluates a TensorFlow/Keras sequential model for text classification.
It tokenizes the text data, builds a Bidirectional LSTM model, compiles it, trains it, and evaluates its performance.
Additionally, it plots the model accuracy and loss trends during training.

detect_anomalies(X_train, y_train, threshold, X_test, y_test):
This function performs anomaly detection on the testing data using a Linear SVC model.
It calculates decision scores for the testing data and identifies anomalies based on a specified threshold.
Anomalies are counted for each day, and a daily trend plot of anomalies is generated.

main():
This is the main function that orchestrates the entire process.
It loads and preprocesses the data, visualizes category distribution, splits the data into train and test sets, trains and evaluates machine learning models, trains and evaluates a TensorFlow/Keras model, detects anomalies, and plots the daily trend of anomalies.

# How to Reproduce the results: 

Environment Setup:
Ensure you have Python installed on your system along with the necessary libraries specified in the script, including numpy, pandas, matplotlib, seaborn, nltk, scikit-learn, and tensorflow.

Data Preparation: # remember the dataset is private

Download the two CSV files, namely "311DataDump_Oct2020_Dec2021.csv" and "311DataDump_10292020_1403.csv", and place them in the same directory as the script.
Make sure the CSV files contain the required columns: 'created_on', 'category', and 'issue_desc'. 

Run the Script:
Open the Python script in your preferred Python environment or text editor.
Execute the script by running the main() function.

Analysis and Results:
The script will load and preprocess the data, visualize the distribution of original categories, split the data into train and test sets, and train and evaluate various machine learning models and a TensorFlow/Keras model.
The accuracy scores for each model will be printed to the console.
Anomaly detection will be performed on the testing data, and the daily trend of anomalies will be plotted.

Interpreting Results:
Observe the accuracy scores of the machine learning models to see their performance in classifying the 311 service requests.
You will see the daily trend of anomalies to identify any unusual patterns or spikes in service requests that deviate from the norm.

Adjusting Parameters:
Experiment with different anomaly detection thresholds to observe their effects on anomaly detection results.
