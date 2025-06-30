import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and Merge Datasets
# Load business and review datasets
business_df = pd.read_csv('yelp_academic_dataset_business.csv')
review_df = pd.read_csv('yelp_academic_dataset_review.csv')

# Merge review data with business info on business_id
merged_df = pd.merge(review_df, business_df, on='business_id', how='inner')
merged_df = merged_df.sort_values(by='business_id')  # Group reviews by business

# Preview merged data
print("Merged Business + Review Data:")
print(merged_df.head(20))



# Summary Statistics on Reviews
# Total number of reviews
total_reviews = len(review_df)

# Number of unique users
unique_users = review_df['user_id'].nunique()

# Top 10 users by number of reviews
top_users = review_df['user_id'].value_counts().head(10)

# Display results
print(f"\nTotal number of reviews: {total_reviews}")
print(f"Number of unique users: {unique_users}")
print("Top 10 users by number of reviews:")
print(top_users)



# Restaurant Category Analysis
# Drop rows with missing category values
business_df = business_df.dropna(subset=['categories'])

# Filter businesses that include 'Restaurants'
restaurant_df = business_df[business_df['categories'].str.contains('Restaurants', na=False)]

# Flatten all categories from restaurant businesses
all_restaurant_categories = restaurant_df['categories'].str.split(', ')
flat_categories = [item for sublist in all_restaurant_categories for item in sublist]

# Count frequency of each restaurant-related category
restaurant_category_counts = pd.Series(flat_categories).value_counts()

# Display and visualize top 20 restaurant categories
print("\nTop 20 Restaurant Categories:")
print(restaurant_category_counts.head(20))

plt.figure(figsize=(12, 8))
restaurant_category_counts.head(20).plot(kind='barh', color='skyblue')
plt.title('Top 20 Restaurant Categories on Yelp')
plt.xlabel('Number of Occurrences')
plt.ylabel('Category')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# All Business Categories Analysis
# Explode all business categories
all_categories = business_df['categories'].str.split(', ').explode()

# Count total category frequency
category_counts = all_categories.value_counts()

# Display and visualize top 20 business categories
print("\nTop 20 Yelp Business Categories:")
print(category_counts.head(20))

top20 = category_counts.head(20).reset_index()
top20.columns = ['Category', 'Count']

plt.figure(figsize=(12, 8))
sns.barplot(data=top20, y='Category', x='Count', palette='Blues_r')
plt.title('Top 20 Yelp Business Categories by Frequency')
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.show()