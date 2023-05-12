import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import umap
import hdbscan


class MPNETopic:
    """
    Take in a dataframe of texts, use sentence transformers to embed the documents, and create a topic model for the documents.
    """
    
    def __init__(self, text_df, column_name, reduce_clusters_to=None):
        """
        Initialize class with dataframe and text column of interest.
        """
        self.text_df = text_df
        self.column_name = column_name
        self.reduce_clusters_to = reduce_clusters_to
        
    def create_embeddings(self, data):
        """
        Use the all-mpnet-base-v2 model to create sentence embeddings for each documents
        """
        model = SentenceTransformer('all-mpnet-base-v2')
        #create sentence embeddings
        embeddings = model.encode(data, show_progress_bar=True)
        return embeddings
    
    def umap_reduction(self, n_neighbors, n_components, embeddings):
        """
        Use umap to reduce dimensionality of the sentence transformer embeddings.
        """
        #reduce dimensionality using umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, #size of the local neighborhood
                            n_components=n_components, #number of parameters to reduce to
                            metric='cosine')
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings
    
    def HDBSCAN_clustering(self, min_cluster_size, umap_embeddings):
        """
        Use HDBSCAN to create clusters from the document embedding. This works well with umap dimensionality reduction.
        """
        #cluster the embeddings with hdbscan
        #hdbscan automatically identifies cluster amount using areas high density with respect to the sentence embeddings
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                                  cluster_selection_method='eom')
        #fit our reduced dimensionality embeddings
        clusters = cluster.fit(umap_embeddings)
        return clusters
    
    def generate_cluster_plot(self, embeddings, clusters):
        """
        Generate a 2-D plot of our clusters from the HDBSCAN clustering.
        """
        #map our sentence embeddings to 2 dimensions
        umap_data = umap.UMAP(n_neighbors=15, n_components=2,
                              min_dist=0.0, metric='cosine').fit_transform(embeddings)
        #store in dataframe and add cluster labels from hdbscan
        results = pd.DataFrame(umap_data, columns=['x', 'y'])
        results['label'] = clusters.labels_

        #plot clusters in 2-D
        fig, ax = plt.subplots(figsize=(20, 6))
        outliers = results.loc[results['label']==-1, :]
        clustered = results.loc[results['label']!=-1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.label, s=0.05, cmap='hsv_r')
        plt.colorbar()
        plt.show()
        
        
    #create our own tfidf function
    def class_based_tf_idf(self, documents, doc_count, ngram_range=(1, 1)):
        """
        Compute class based tf-idf to supply documents within the same label or class (cluster) the same class vector.
        """
        #ngram count from documents
        count = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(documents)

        #term frequency by cluster
        t = count.transform(documents).toarray()
        #total number of words per topic
        w = t.sum(axis=1)
        #term frequency divided by word count
        tf = np.divide(t.T, w)
        #total word frequency of entire corpus
        sum_t = t.sum(axis=0)

        #inverse document frequency
        idf = np.log(np.divide(doc_count, sum_t)).reshape(-1, 1)
        #multiple term frequency by inverse document frequency
        tf_idf = np.multiply(tf, idf)
        return tf_idf, count
    
    def topic_top_n_words(self, tfidf, fit_vectorizer, topic_df, n=20):
        """
        Extract top n most important words per topic based on class tfidf vector.
        """
        #get word labels from vectorizer
        words = fit_vectorizer.get_feature_names_out()

        #list of cluster labels (topics)
        labels = list(topic_df['Topic'])
        #transpose tfidf to represent topics in rows and word score in columns
        tfidf_trans = tfidf.T
        #return sorted indices based on importance (low to high) and take last n
        indices = tfidf_trans.argsort()[:, -n:]

        #return dictionary with topic label and the top n words per topic with importance score
        top_n_words = {label: [(words[j], tfidf_trans[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def topic_sizes(self, df):
        """
        Count of articles per topic in a dataframe.
        """
        topic_size = df.groupby(['Topic'], as_index=False)[self.column_name].count()\
                       .rename(columns={self.column_name:'Size'}).sort_values('Size', ascending=False)
        return topic_size
    
    def reduce_topic_dimensionality(self, reduce_clusters_to, tfidf, countvec, text_dataframe):
        """
        Combine the smallest cluster with its highest similarity cluster and then recreate the topic model.
        Iterate X times until we have removed a set number of clusters.
        """
        #number of times to iterate for desired topic number
        clusters_to_remove = text_dataframe['Topic'].nunique() - reduce_clusters_to
        
        for i in range(clusters_to_remove):
            #compute similarity between each topic
            similarities = cosine_similarity(tfidf.T)
            #make the diagonal of the make 0, (set topic similarity with itself to 0)
            np.fill_diagonal(similarities, 0)

            #get topic size
            topic_size = topic_sizes(text_dataframe)

            #index of smallest topic
            topic_being_merged = topic_size.iloc[-1]['Topic']
            #highest similarity topic
            topic_merged_to = np.argmax(similarities[topic_being_merged + 1]) - 1

            #adjust the dataframe topics
            text_dataframe['Topic'] = [topic_merged_to if top == topic_being_merged else top for top in text_dataframe['Topic']]

            #create topic map to change topic numbers
            old_topics = text_dataframe['Topic'].sort_values().unique()
            topic_map = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
            #map to new topic numbers
            text_dataframe['Topic'] = text_dataframe['Topic'].map(topic_map)
            #aggregate document text by topic again
            all_topic_text = text_dataframe.groupby(['Topic'], as_index=False).agg({self.column_name: ' '.join})

            #recreate topic model
            doc_count = len(text_dataframe)
            tfidf, countvec = class_based_tf_idf(all_topic_text[self.column_name].values, doc_count)
            top_n_words = topic_top_n_words(tfidf, countvec, all_topic_text, n=20)
        return text_dataframe, tfidf, countvec, top_n_words
    
    def create_topic_model(self):
        """
        Create topic model from input text dataframe.
        """
        avg_length = self.text_df[self.column_name].apply(lambda article: len(article)).mean()
        print(f"Average title word length: {int(avg_length):,}\n")
        
        #subset column we are interested in
        data = self.text_df[self.column_name]
        
        #create sentence embeddings
        embeddings = self.create_embeddings(data)
        
        #reduce dimensionality
        umap_embeddings = self.umap_reduction(n_neighbors=15, n_components=10, embeddings=embeddings)
        #cluster documents
        clusters = self.HDBSCAN_clustering(min_cluster_size=15, umap_embeddings=umap_embeddings)
        initial_clust = len(set(clusters.labels_))
        
        #generate 2-D plot
        self.generate_cluster_plot(embeddings=embeddings, clusters=clusters)
        
        my_text_df = self.text_df.loc[:]
        #add cluster labes to document dataframe
        my_text_df['Topic'] = clusters.labels_
        #create document id key
        my_text_df['ID'] = range(len(my_text_df))
        #groupby cluster and combine all article text in one row
        all_topic_text = my_text_df.groupby(['Topic'], as_index=False).agg({self.column_name: ' '.join})

        #calculate tfidf and total count
        tf_idf, cv = self.class_based_tf_idf(all_topic_text[self.column_name].values, doc_count=len(data))
        
        #most important words per cluster (topic)
        top_n_words = self.topic_top_n_words(tf_idf, cv, all_topic_text, n=20)
        #size of each topic
        size = self.topic_sizes(my_text_df)

        #get largest 10 topics (if there are 10) excluding noise (-1)
        top_topics = size.loc[size['Topic']!=-1][:10]['Topic'].tolist()
        
        print(f"Initial Topic Amount: {initial_clust:,}")
        if self.reduce_clusters_to is not None:
            new_text_df, tfidf, cv, top_n_words = self.reduce_topic_dimensionality(20, tf_idf, cv, my_text_df)
            print(f"Final Topic Amount: {len(top_n_words):,}\n")
            return top_n_words
        else:
            print(f"Final Topic Amount: {len(top_n_words):,}\n")
            return top_n_words