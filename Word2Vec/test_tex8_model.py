from gensim.models import word2vec

# Load model
model = word2vec.Word2Vec.load('/tmp/text8.model')

# Get results
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
