import pandas as pd
# splitting the text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('yelp_review_business_sample_cleaned.csv')
# Create binary sentiment labels from star ratings:
# Reviews with 1-2 stars are labeled as negative (0), 4-5 stars as positive (1); 3-star reviews are dropped
df = df[df['stars_x'] != 3]
df['Target_Label'] = df['stars_x'].apply(lambda x: 1 if x >= 4 else 0)
train, test = train_test_split(df, test_size=0.2, random_state=42)
tv = TfidfVectorizer(max_features=10000)
article_train = tv.fit_transform(train['cleaned_text'])
article_test = tv.transform(test['cleaned_text'])

# Text classification
# Trying to find which classifier has higher accuracy
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=300, random_state=42),
    'SVM (LinearSVC)': LinearSVC(),
    'SVM (SGDClassifier)': SGDClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 50, 50), random_state=42,
                                     solver='adam', learning_rate='adaptive', activation='relu')
}

# Store evaluation results (accuracy and training time)
results = []

# Loop through each model, train it, and evaluate its performance
for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    model.fit(article_train, train['Target_Label'])  # Train the model
    predictions = model.predict(article_test)        # Predict on test data
    accuracy = accuracy_score(test['Target_Label'], predictions)  # Calculate accuracy
    cm = confusion_matrix(test['Target_Label'], predictions)      # Generate confusion matrix
    elapsed = time.time() - start_time
    results.append((name, accuracy, elapsed))
    print(f"{name} accuracy: {accuracy:.4f}, Time: {elapsed:.2f} sec")
    print(f"Confusion Matrix:\n{cm}")

# Compile results into a summary DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Time (sec)'])

# Print the final model comparison table
print("\nModel Performance Comparison:")
print(results_df)


