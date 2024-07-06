import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."""

stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
tokens = [token.text for token in doc]
word_freq = {}

for word in doc:
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        if word.text not in word_freq.keys():
            word_freq[word.text] = 1
        else:
            word_freq[word.text] += 1

max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq
sent_tokens = [sent for sent in doc.sents]
sent_scores = {}
for sent in sent_tokens:
    for word in sent:
        if word.text in word_freq.keys():
            if sent not in sent_scores.keys():
               sent_scores[sent] = word_freq[word.text]
            else:
                sent_scores[sent] +=word_freq[word.text]
select_len = int(len(sent_tokens) * 0.3)
summary = nlargest(select_len,sent_scores,key = sent_scores.get)
final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)
print(text)
print("Length of Original text",len(text.split(' ')))
print(summary)
print("Length of summary text",len(summary.split(' ')))


