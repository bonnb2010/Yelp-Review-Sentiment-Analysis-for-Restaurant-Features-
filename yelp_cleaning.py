import nltk
import pandas as pd
import spacy
import text_normalizer as tn

# defines
# If you think of additional restriction words, just add them in
DIETARY_LIST = ["gluten","dairy","vegan","celiac",
                "vegetarian","allergy","nut",
                "halal","kosher","lactose","plant",
                "diabetes","veg","diabetic","carb",
                "allergies"]

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

yelp_rev = pd.read_csv('yelp_academic_dataset_review.csv')
yelp_business = pd.read_csv('yelp_academic_dataset_business.csv')

# merge business data to review data
yelp_rev['business_id'] = yelp_rev['business_id'].astype(str)
yelp_business['business_id'] = yelp_business['business_id'].astype(str)

yelp_rev_bus = yelp_rev.merge(yelp_business, on='business_id', how='left')

# make sure we are only looking at restaurants
yelp_rev_bus = yelp_rev_bus[yelp_rev_bus['categories'].str.contains('Restaurants', na=False)]

# regex to make sure we are only sampling reviews that mention dietary restrictions
pattern = '|'.join(DIETARY_LIST)
yelp_rev_bus = yelp_rev_bus[yelp_rev_bus['text'].str.contains(pattern)]

yelp_rev_bus['cleaned_text'] = tn.normalize_corpus(yelp_rev_bus['text'], html_stripping=True,
                                        contraction_expansion=True,
                                        accented_char_removal=True, text_lower_case=True,
                                        text_lemmatization=True, special_char_removal=True,
                                        stopword_removal=True, misspelled_words_correction=False)

print(yelp_rev_bus.shape[0])

yelp_rev_bus.to_csv('./yelp_review_business_sample_cleaned.csv')
