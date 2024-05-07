import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Download NLTK stop words data
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

# Function to load and preprocess data
def load_and_preprocess_data():
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "311DataDump_Oct2020_Dec2021.csv")
    file_path_2 = os.path.join(current_directory, "311DataDump_10292020_1403.csv")
    
    data = pd.read_csv(file_path)
    data_2 = pd.read_csv(file_path_2)

    # Dropping unnecessary columns
    columns_to_drop = ['cl_full_name', 'cl_first_name', 'cl_last_name', 'cl_phone',
                      'full_address', 'house_number', 'street_dir', 'street_name',
                      'street_suffix', 'street_type', 'unit','parcel_id','pw_facility', 
                      'ce_zone', 'latitude', 'longitude', 'client_type', 'client_type_detail',
                      'city','intersection', 'location', 'jurisdiction', 'bcc','status_code','activity_type']
    data = data.drop(columns=columns_to_drop)
    data_2 = data_2.drop(columns=columns_to_drop)

    # Concatenating dataframes
    combined_df = pd.concat([data_2, data])

    # Selecting relevant columns
    filtered_df = combined_df[['created_on','category','sub_category','sub_category_detail','issue_desc','issue_resol']]

    return filtered_df

# Function to preprocess features and labels
def preprocess_features_and_labels(filtered_df):
    X = filtered_df['issue_desc'].apply(preprocess_text)
    y = filtered_df['category']
    
    # Remove missing values
    data = pd.DataFrame({'X': X, 'y': y})
    data = data.dropna(subset=['y'])
    data = data[data['X'] != "requested information contact"]
    data = data[data['X'].str.strip() != ""]
    
    X = data['X']
    y = data['y']
    
    return X, y

# Function to visualize original categories distribution
def visualize_original_categories_distribution(data):
    original_categories = data['y'].unique()
    total_requests = len(data)
    category_distribution = {}
    for category in original_categories:
        category_data = data[data['y'] == category]
        category_count = len(category_data)
        category_ratio = category_count / total_requests
        category_distribution[category] = {'count': category_count, 'ratio': category_ratio}

    categories = list(category_distribution.keys())
    counts = [stats['count'] for stats in category_distribution.values()]
    ratios = [stats['ratio'] for stats in category_distribution.values()]

    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.35
    bars1 = ax.bar(categories, counts, bar_width, label='Request Count', color='blue')
    ax2 = ax.twinx()
    line2 = ax2.plot(categories, ratios, label='Request Ratio', color='orange', marker='o')

    ax.set_xticklabels(categories, rotation=90, ha='center', fontsize=20)
    ax2.set_xticklabels(categories, rotation=90, ha='center', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)

    ax.set_xlabel('Original Categories', fontsize=20)
    ax.set_ylabel('Request Count', fontsize=20)
    ax2.set_ylabel('Request Ratio',  fontsize=20)
    ax.legend(loc='upper left', fontsize=20)
    ax2.legend(loc='upper right', fontsize=20)

    ax.set_facecolor('white')
    ax2.set_facecolor('white')

    ax.grid(False)
    ax2.grid(False)

    plt.show()

# Function to train and evaluate models
def train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    # Linear SVM
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    y_pred_rf = rf_model.predict(X_test_tfidf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Decision Tree
    DT_model = DecisionTreeClassifier(random_state=42)
    DT_model.fit(X_train_tfidf, y_train)
    y_pred_rf = DT_model.predict(X_test_tfidf)
    accuracy_DT = accuracy_score(y_test, y_pred_rf)
    
    
    # Naive Bayes
    naive_model = GaussianNB()
    naive_model.fit(X_train_tfidf, y_train)
    y_pred_naive = naive_model.predict(X_test_tfidf)
    accuracy_naive = accuracy_score(y_test, y_pred_naive)
    
    return accuracy_svm, accuracy_rf, accuracy_DT, accuracy_naive

# Function to train and evaluate the TensorFlow/Keras model
def train_keras_model(X_train, y_train, X_test, y_test):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = pad_sequences(X_test_seq, maxlen=100)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=100),
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),  # Adding L2 regularization
        Dropout(0.5),  # Adjusted dropout rate
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Adding L2 regularization
        Dense(len(encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_pad, y_train_encoded, epochs=10, batch_size=32,
        validation_split=0.1, callbacks=[early_stopping]
    )

    loss, accuracy = model.evaluate(X_test_pad, y_test_encoded)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# Function for anomaly detection and plotting
def detect_anomalies(X_train, y_train, threshold, X_test, y_test):
    # Train the Linear SVC model
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    
    # Calculate decision scores for the testing data
    decision_scores_test = svm_model.decision_function(X_test)
    
    # Get the anomalies
    anomalies = np.max(decision_scores_test) < threshold
    
    # Count anomalies for each day
    anomalies_counts = X_test['created_on'].groupby(pd.to_datetime(X_test['created_on']).dt.date).agg(lambda x: anomalies[x].sum())
    
    # Plot daily trend of anomalies
    plt.figure(figsize=(10, 6))
    anomalies_counts.plot(marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Anomaly Count')
    plt.title('Daily Trend of Anomalies')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return anomalies_counts

# Main function
def main():
    # Load and preprocess data
    filtered_df = load_and_preprocess_data()
    
    # Preprocess features and labels
    X, y = preprocess_features_and_labels(filtered_df)
    
    # Visualize original categories distribution
    visualize_original_categories_distribution(filtered_df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train and evaluate models
    accuracy_svm, accuracy_rf, accuracy_DT, accuracy_naive = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    print("Linear SVM Accuracy:", accuracy_svm)
    print("Random Forest Accuracy:", accuracy_rf)
    print("Decision Tree Accuracy:", accuracy_DT)
    print("Naive Bayes Accuracy:", accuracy_naive)
    
    # Train and evaluate the TensorFlow/Keras model
    train_keras_model(X_train, y_train, X_test, y_test)
    
    
    # Set the threshold for anomaly detection
    threshold = 0 
    
    # Detect anomalies and plot daily trend
    anomalies_counts = detect_anomalies(X_train, y_train, threshold, X_test, y_test)
    print("Anomalies counts for each day:")
    print(anomalies_counts)

if __name__ == "__main__":
    main()
