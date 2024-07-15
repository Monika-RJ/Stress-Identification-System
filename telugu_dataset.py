SVM classifier

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
# Load your dataset (replace 'data.csv' with your dataset)
data = pd.read_csv('/content/drive/MyDrive/TELUGU_TRAINING_DATA.csv')

# Assuming your dataset has 'text' column for news text and 'label' column for labels (1 for fake, 0 for real)
X = data['TEXT DATA']
y = data['LABEL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train a Support Vector Machine (SVM) classifier (you can use other classifiers as well)
classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, y_train)

# Predict labels on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the classifier
accuracysvm = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracysvm)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

# Example usage on a new test text (replace 'test_text' with your actual test text)
test_text = pd.read_csv("/content/drive/MyDrive/full_telugu_data_test - full_telugu_data_test.csv")

# Preprocess the test text and transform it using the same vectorizer
test_text_tfidf = vectorizer.transform(test_text['Text data'])

# Predict the labels for the test text
test_predictions = classifier.predict(test_text_tfidf)

for i, prediction in enumerate(test_predictions):
    label = "stressed" if prediction == 1 else "Non stressed"
    print(f"Test {i+1}: {test_text['Text data'].iloc[i]} - Predicted: {label}")

for true_label, pred_label in zip(y_test[:1050], test_predictions[:1050]):
    print(f"True Label: {true_label}, Predicted Label: {pred_label}")

import pandas as pd

predicted_labels = test_predictions[:1050]
results_df = pd.DataFrame({
    'Test Text': test_text['Text data'],
    'Predicted_Labels': predicted_labels})

# Save the results to a CSV file
results_df.to_csv('/content/drive/MyDrive/Telugu_SVC_01.tsv', index=False)

print("\nResults saved to Telugu_SVC_01.tsv")

------------------------------------------------------------------------------------------------------------

Random Forest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing

# Load your dataset (replace 'data.csv' with your dataset)
data = pd.read_csv('/content/drive/MyDrive/TELUGU_TRAINING_DATA.csv')

# Assuming your dataset has 'text' column for news text and 'label' column for labels (1 for fake, 0 for real)
X = data['TEXT DATA']
y = data['LABEL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Predict labels on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the classifier
accuracyrf = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)


# Example usage on a new test text (replace 'test_text' with your actual test text)
test_text = pd.read_csv("/content/drive/MyDrive/full_telugu_data_test - full_telugu_data_test.csv")

# Preprocess the test text and transform it using the same vectorizer
test_text_tfidf = vectorizer.transform(test_text['Text data'])

# Predict the labels for the test text
test_predictions = classifier.predict(test_text_tfidf)

for i, prediction in enumerate(test_predictions):
    label = "stressed" if prediction == 1 else "Non stressed"
    print(f"Test {i+1}: {test_text['Text data'].iloc[i]} - Predicted: {label}")

for true_label, pred_label in zip(y_test[:1050], test_predictions[:1050]):
    print(f"True Label: {true_label}, Predicted Label: {pred_label}")


print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{report}")

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title("Random Forest")
plt.show()
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

import pandas as pd

predicted_labels = test_predictions[:1050]
results_df = pd.DataFrame({
    'Test Text': test_text['Text data'],
    'Predicted_Labels': predicted_labels})

# Save the results to a CSV file
results_df.to_csv('/content/drive/MyDrive/Telugu_RF_01.tsv', index=False)

print("\nResults saved to Telugu_RF_01.csv")

---------------------------------------------------------------------------------------------

Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing

# Load your dataset (replace 'data.csv' with your dataset)
data = pd.read_csv('/content/drive/MyDrive/TELUGU_TRAINING_DATA.csv')

# Assuming your dataset has 'text' column for news text and 'label' column for labels (1 for fake, 0 for real)
X = data['TEXT DATA']
y = data['LABEL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the performance of the model
accuracynb = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

# Example usage on a new test text (replace 'test_text' with your actual test text)
test_text = pd.read_csv("/content/drive/MyDrive/full_telugu_data_test - full_telugu_data_test.csv")

# Preprocess the test text and transform it using the same vectorizer
test_text_tfidf = vectorizer.transform(test_text['Text data'])

# Predict the labels for the test text
test_predictions = classifier.predict(test_text_tfidf)

for i, prediction in enumerate(test_predictions):
    label = "stressed" if prediction == 1 else "Non stressed"
    print(f"Test {i+1}: {test_text['Text data'].iloc[i]} - Predicted: {label}")

for true_label, pred_label in zip(y_test[:1050], test_predictions[:1050]):
    print(f"True Label: {true_label}, Predicted Label: {pred_label}")

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix}")
print(f"Classification Report:\n{classification_rep}")

import pandas as pd

predicted_labels = test_predictions[:1050]
results_df = pd.DataFrame({
    'Test Text': test_text['Text data'],
    'Predicted_Labels': predicted_labels})

# Save the results to a CSV file
results_df.to_csv('/content/drive/MyDrive/Telugu_NB_01.tsv', index=False)

print("\nResults saved to Telugu_NB_01.csv")

import numpy as np
y= [accuracysvm,accuracyrf,accuracynb]
x=["SVM","Random forest","Naive Bayes"]
c = ['orange', 'blue', 'olive']
fig, ax = plt.subplots()
bars = ax.bar(x, y, color=c, width=0.4) # Adjust the bar width

# Add labels and title
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Algorithm vs Accuracy')

# Adjust the y-axis limits to increase the space for labels
plt.ylim(0, 1.1) # Adjusted to provide space for labels

# Display the values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

# Set y-axis ticks with three decimal places
plt.yticks(np.arange(0, max(y) + 0.1, 0.1))
plt.show()
