import json
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding


VOCAB_PATH = 'model/vocabulary.json'
PRETRAINED_PATH = 'model/pytorch_model.bin'

TEST_SENTENCES = ['I love mom\'s cooking',
                  'I love how you never reply back..',
                  'I love cruising with my homies',
                  'I love messing with yo mind!!',
                  'I love you and now you\'re just gone..',
                  'This is shit',
                  'This is the shit']


maxlen = 30
batch_size = 32

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_feature_encoding(PRETRAINED_PATH)
print(model)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
encoding = model(tokenized)

avg_across_sentences = np.around(np.mean(encoding, axis=0)[:5], 3)

print('printing results')
print(avg_across_sentences)

assert np.allclose(avg_across_sentences, np.array([-0.023, 0.021, -0.037, -0.001, -0.005]))