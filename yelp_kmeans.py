import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import text_normalizer as tn
import seaborn as sns

file = 'yelp_review_business_sample_cleaned.csv'
df = pd.read_csv(file)
df = df.sample(n=2000, random_state=42)

# K-means optimal clusters with minibatch
tfidf = TfidfVectorizer(stop_words='english', max_features=10000, min_df=5, max_df=0.7)
df = df[df['cleaned_text'].notna()]
text = tfidf.fit_transform(df['cleaned_text'])

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 1)
    sse=[]
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=42).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    f, ax = plt.subplots()
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()

find_optimal_clusters(text, 20)

# Based on the elbow method, K=7 was identified as the most appropriate choice for K-means clustering.
# K-means interpretation
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    df_terms = pd.DataFrame()
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join(labels[t] for t in np.argsort(r)[-n_terms:]))
        df_terms[i] = [labels[t] for t in np.argsort(r)[-n_terms:]]
    print(df_terms)
    savefile = "k_means" + str(len(df)) + '_topics_keywords.csv'
    df_terms.to_csv(savefile, index=False)
clusters = KMeans(n_clusters=7, random_state=42).fit_predict(text)
get_top_keywords(text, clusters, tfidf.get_feature_names_out(), 10)
df['kmeans_cluster'] = clusters

print(df['kmeans_cluster'])


