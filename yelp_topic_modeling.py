import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt

# Convert cleaned text to tokens
def sentence_to_words(texts):
    for line in texts:
        yield gensim.utils.simple_preprocess(str(line), deacc=True)

# Optimize number of topics
def coherence_test(text_series, start_topics=2, end_topics=10):
    # Preprocessing
    data_words = list(sentence_to_words(text_series.dropna().values.tolist()))

    # Create dictionary and corpus
    id2word = corpora.Dictionary(data_words)
    id2word.filter_extremes(no_below=10, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in data_words]

    # Store scores and models
    coherence_scores = []
    models_list = []

    for num_topics in range(start_topics, end_topics + 1):
        print(f"Training LDA with {num_topics} topics...")
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=42,
                                           update_every=1,
                                           chunksize=200,
                                           passes=5,
                                           alpha='auto',
                                           per_word_topics=True)
        coherence_model = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model.get_coherence()
        coherence_scores.append(coherence_lda)
        models_list.append(lda_model)
        print(f"Coherence Score for {num_topics} topics: {coherence_lda:.4f}")

    # Find the best number of topics
    best_index = coherence_scores.index(max(coherence_scores))
    best_num_topics = start_topics + best_index
    best_model = models_list[best_index]

    # Print the best topic result
    print(f"\nBest Number of Topics: {best_num_topics} with Coherence = {coherence_scores[best_index]:.4f}")
    pprint(best_model.print_topics())

    return best_num_topics, corpus, id2word, best_model

# Main execution
def main():
    file = 'yelp_review_business_sample_cleaned.csv'
    df = pd.read_csv(file)
    df = df.sample(n=2000, random_state=42)
    num_topics, corpus, id2word, model = coherence_test(df['cleaned_text'], start_topics=2, end_topics=10)

if __name__ == "__main__":
    main()