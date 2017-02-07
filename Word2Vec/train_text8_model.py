from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# load corpus
sentences = word2vec.Text8Corpus('/tmp/text8')

# train the skip-gram model
model = word2vec.Word2Vec(sentences, size=200)

# pickle the model to disk so that we can resume training later
model.save('/tmp/text8.model')

# store the learned weights in a format that the original C tool understands
model.save_word2vec_format('/tmp/text8.model.bin', binary=True)

# or import word vectors created by the C word2vec
# model = word2vec.Word2Vec.load_word2vec_format('/tmp/temp8.model.bin',
#                                                binary=True)
