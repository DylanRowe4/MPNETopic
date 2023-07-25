# MPNETopic
MPNETopic uses sentence transformers from Hugging Face to create sentence embeddings and then a subsequent topic model for documents in a corpus.

This is done by using UMAP to reduce the dimensionality of our embeddings and HDBSCAN to then identify clusters of high density in our document feature space. Fundamentally, UMAP works by constructing a high-dimensional graph representation of the input embeddings from our sentence transformer and then creates an optimal low-dimensional graph to be as structurally similar as possible. HDBSCAN then takes this low-dimensional graph, estimates areas of high density in the points of our feature space and then combines them into selected groups or clusters. The clusters then form the basis or "topics" for our topic model. Once the topics are identified an in-topic TFIDF analysis can be conducted to identify the most important words or phrases in each topic.