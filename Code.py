import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import re 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 

# Download NLTK stop words data
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

#************************************************************************************************

# Get the current working directory where your script is located
current_directory = os.getcwd()

# Correctly combine the directory and file name
file_path = os.path.join(current_directory, "311DataDump_Oct2020_Dec2021.csv")

file_path_2 = os.path.join(current_directory, "311DataDump_10292020_1403.csv")
# Read the CSV file
data = pd.read_csv(file_path)
data_2 = pd.read_csv(file_path_2)


#Droping certain columns we dont need the


data = data.drop(['cl_full_name', 'cl_first_name', 'cl_last_name', 'cl_phone',
'full_address', 'house_number', 'street_dir', 'street_name',
'street_suffix', 'street_type', 'unit','parcel_id','pw_facility', 
'ce_zone', 'latitude', 'longitude', 'client_type', 'client_type_detail',
'city','intersection', 'location', 'jurisdiction', 'bcc','status_code','activity_type'], axis=1)


data_2 = data_2.drop(['cl_full_name', 'cl_first_name', 'cl_last_name', 'cl_phone',
'full_address', 'house_number', 'street_dir', 'street_name',
'street_suffix', 'street_type', 'unit','parcel_id','pw_facility', 
'ce_zone', 'latitude', 'longitude', 'client_type', 'client_type_detail',
'city','intersection', 'location', 'jurisdiction', 'bcc','status_code','activity_type'], axis=1)


# Concatenate them
combined_df = pd.concat([data_2, data])


# Assuming 'date_time' is the column with your datetime information
# Convert 'date_time' to datetime if it's not already in that format
combined_df['created_on'] = pd.to_datetime(combined_df['created_on'])


filtered_df = combined_df[['created_on','category','sub_category','sub_category_detail','issue_desc',
                                  'issue_resol']]


###########################################################################################

X = filtered_df['issue_desc'].apply(preprocess_text)
y = filtered_df['category']

# Check for missing data in X
missing_data_X = X.isnull().sum()
print("Missing data in X:")
print(missing_data_X)

# Check for missing data in y (assuming y is a Series)
missing_data_y = y.isnull().sum()
print("\nMissing data in y:")
print(missing_data_y)

# Combine X and y into a DataFrame (if not already)
data = pd.DataFrame({'X': X, 'y': y})

# Check for missing data
missing_data = data.isnull().sum()
print("Missing data before removal:")
print(missing_data)

# Remove rows with missing values in y
data = data.dropna(subset=['y'])

# Remove rows where X is "requested information contact"
data = data[data['X'] != "requested information contact"]

# Remove rows where any cell in X contains empty words
data = data[data['X'].str.strip() != ""]

# Separate X and y again
X = data['X']
y = data['y']

# Check for missing data after removal
missing_data_after_removal = data.isnull().sum()
print("\nMissing data after removal:")
print(missing_data_after_removal)

#********************************************************

# Explore the distribution of requests across the original categories
original_categories = data['y'].unique()
#original_categories = filtered_df['category'].unique()
total_requests = len(data)

category_distribution = {}
for category in original_categories:
    category_data = data[data['y'] == category]
    #category_data = filtered_df['category'][filtered_df['category'] == category]
    category_count = len(category_data)
    category_ratio = category_count / total_requests
    category_distribution[category] = {'count': category_count, 'ratio': category_ratio}

# Display the results
print("Original Categories and Request Distribution:")
for category, stats in category_distribution.items():
    print(f"{category}: {stats['count']} requests ({stats['ratio']*100:.2f}%)")
    
    
# Extract category names, counts, and ratios for plotting
categories = list(category_distribution.keys())
counts = [stats['count'] for stats in category_distribution.values()]
ratios = [stats['ratio'] for stats in category_distribution.values()]

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
bar_width = 0.35

# Bar chart for counts
bars1 = ax.bar(categories, counts, bar_width, label='Request Count', color='blue')

# Line chart for ratios
ax2 = ax.twinx()
line2 = ax2.plot(categories, ratios, label='Request Ratio', color='orange', marker='o')

# Rotating labels for better visibility and increasing font size
ax.set_xticklabels(categories, rotation=90, ha='center', fontsize=20)
ax2.set_xticklabels(categories, rotation=90, ha='center', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax2.tick_params(axis='both', labelsize=20)

# Adding labels and title with increased font size
ax.set_xlabel('Original Categories', fontsize=20)
ax.set_ylabel('Request Count', fontsize=20)
ax2.set_ylabel('Request Ratio',  fontsize=20)
ax.legend(loc='upper left', fontsize=20)
ax2.legend(loc='upper right', fontsize=20)

# Set background color to white
ax.set_facecolor('white')
ax2.set_facecolor('white')

# Remove grids
ax.grid(False)
ax2.grid(False)

# Display the plot
plt.show()



#*******************************************************


# 2. Data Preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Feature Extraction
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 4. Model Training
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)


# 5. Prediction
y_pred = svm_model.predict(X_test_tfidf)
y_pred_series = pd.Series(y_pred)

print(y_pred_series.value_counts())
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Evaluate precision
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Precision:", precision)

# Evaluate recall
recall = recall_score(y_test, y_pred, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Recall:", recall)

# Evaluate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("F1 Score:", f1)

# Evaluate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

#**********************************************************************************
# Assume X and y are your features and labels
# Define the fractions of the dataset to use for training
training_sizes = np.arange(0.05, 1.0, 0.05)

# Lists to store results
train_f1_scores = []
test_f1_scores = []

# Split the data into a fixed test set
X_train_fixed, X_test, y_train_fixed, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_fixed)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model and evaluate on different training set sizes
for size in training_sizes:
    # Use a fraction of the training data
    X_train, _, y_train, _ = train_test_split(X_train_tfidf, y_train_fixed, train_size=size, random_state=42)

    # Create and train your model (replace LinearSVC() with your actual model)
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)

    # Make predictions on the training set
    y_train_pred = svm_model.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_f1_scores.append(train_f1)

    # Make predictions on the test set
    y_test_pred = svm_model.predict(X_test_tfidf)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_f1_scores.append(test_f1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(training_sizes * 100, train_f1_scores, marker='o', label='Training F1-score')
plt.plot(training_sizes * 100, test_f1_scores, marker='o', label='Testing F1-score')
plt.xlabel('Percentage of Training Set Size')
plt.ylabel('F1 score')
#plt.title('Effect of Training Set Size on F1 score')
plt.legend()
plt.grid(True)
plt.show()

#***********************************************************************************

# Nonlinear SVM

from sklearn.svm import SVC

# Using RBF kernel for nonlinear SVM
svm_nonlinear_model = SVC(kernel='rbf')
svm_nonlinear_model.fit(X_train_tfidf, y_train)

# Prediction
y_pred_nonlinear = svm_nonlinear_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy_nonlinear = accuracy_score(y_test, y_pred_nonlinear)
print("Nonlinear SVM Accuracy:", accuracy_nonlinear)


# Using RBF kernel for nonlinear SVM
svm_nonlinear_model = SVC(kernel='rbf')
svm_nonlinear_model.fit(X_train_tfidf, y_train)

# Prediction
y_pred_nonlinear = svm_nonlinear_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy_nonlinear = accuracy_score(y_test, y_pred_nonlinear)
print("Nonlinear SVM Accuracy:", accuracy_nonlinear)
#**********************************************************************

# Naive base (baseline)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)

# Prediction
y_pred_naive = naive_bayes_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy_Naive = accuracy_score(y_test, y_pred_naive)
print("Naive Accuracy:", accuracy_Naive)

# Evaluate precision
precision = precision_score(y_test, y_pred_naive, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Precision:", precision)

# Evaluate recall
recall = recall_score(y_test, y_pred_naive, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Recall:", recall)

# Evaluate F1 score
f1 = f1_score(y_test, y_pred_naive, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("F1 Score:", f1)

# Evaluate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_naive)
print("Confusion Matrix:")
print(conf_matrix)




#***************************************************************************************


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

# Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_tfidf, y_train)

# Prediction
y_pred_dt = decision_tree_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Evaluate precision
precision = precision_score(y_test, y_pred_dt, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Precision:", precision)

# Evaluate recall
recall = recall_score(y_test, y_pred_dt, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Recall:", recall)

# Evaluate F1 score
f1 = f1_score(y_test, y_pred_dt, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("F1 Score:", f1)

# Evaluate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


#****************************************************************************************


# Random Forrest 

from sklearn.ensemble import RandomForestClassifier

# Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_tfidf, y_train)

# Prediction
y_pred_rf = random_forest_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)


# Evaluate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_rf)
print("Decision Tree Accuracy:", accuracy_dt)

# Evaluate precision
precision = precision_score(y_test, y_pred_rf, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Precision:", precision)

# Evaluate recall
recall = recall_score(y_test, y_pred_rf, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("Recall:", recall)

# Evaluate F1 score
f1 = f1_score(y_test, y_pred_rf, average='weighted')  # Use 'micro', 'macro', or 'weighted'
print("F1 Score:", f1)

# Evaluate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(conf_matrix)

##############################################################################################

accuracy_linear_svm = accuracy_score(y_test, y_pred)  # For your linear SVM
accuracy_dt = accuracy_score(y_test, y_pred_dt)  # For your Decision Tree
accuracy_rf = accuracy_score(y_test, y_pred_rf)  # For your Random Forest


import matplotlib.pyplot as plt

# Model names
models = ['Linear SVM', 'Decision Tree', 'Random Forest']

# Accuracy scores
accuracies = [accuracy_linear_svm, accuracy_dt, accuracy_rf]

# Creating the bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])

plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])  # Assuming accuracy is between 0 and 1
plt.xticks(rotation=45)  # Rotates labels to make them readable
plt.grid(axis='y')

# Adding the accuracy values on top of the bars
for i in range(len(models)):
    plt.text(i, accuracies[i] + 0.01, f"{accuracies[i]:.2f}", ha = 'center')

plt.show()

#######################################################################################

# Normalize the confusion matrix
conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

import seaborn as sns

# Create a heatmap
sns.set(font_scale=1.2)  # Adjust the font size if needed
plt.figure(figsize=(10, 8))

# Use normalized confusion matrix for display
heatmap = sns.heatmap(conf_matrix_norm, annot=False, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar_kws={'label': 'Proportion'})

# Adjust color bar font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Adding labels and title
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Normalized Confusion Matrix', fontsize=16)

# Display the plot
plt.show()


#*************************************************

