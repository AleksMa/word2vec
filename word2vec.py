import collections
import numpy as np
import copy
import re
import time

import torch
import gensim


# Утилиты текстовые

def text2words(source, min_token_size=0):
    return [i.lower() for i in re.sub('[^a-zA-Z]',' ', source).split(' ') if len(i) > min_token_size]

def words2ids(words, vocabulary):
    return [vocabulary[token] for token in words if token in vocabulary]

def words2vocabulary(words, max_size=1000000, padding_word=None):
    word_counts = collections.defaultdict(int)

    for token in words:
        word_counts[token] += 1

    sorted_word_counts = [(padding_word, 0)]+ sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    vocabulary = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    return vocabulary

# Iterable dataset батчей - окон слов

class WordsDataset(torch.utils.data.Dataset):
    def __init__(self, words, targets, out_len=100, pad_value=0):
        self.words = words
        self.targets = targets
        self.out_len = out_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        txt = [self.words[i] for i in range(item - self.out_len, item)]
        txt = torch.tensor(txt, dtype=torch.long)

        return txt, torch.tensor(0, dtype=torch.long)

# Собственно нейросеть

class NN(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, sentence_len, radius=5, negative_samples_n=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.negative_samples_n = negative_samples_n
        self.embeddings = torch.nn.Embedding(self.vocab_size, emb_size, padding_idx=0)

        # двухдиагональная матрица с полосами ширины radius из единиц и нулями на главной диагонали
        self.positive_sim_mask = sum([torch.diag(torch.ones(sentence_len), diagonal=i)[:-abs(i), :-abs(i)] for i in range(-radius, radius+1) if i != 0])
    
    def forward(self, batch):
        batch_size = batch.shape[0]          # batch: BatchSize x SentSize 
        batch_embs = self.embeddings(batch)   # BatchSize x SentSize x EmbSize

        # SkipGram
        positive_embs = batch_embs.permute(0, 2, 1)                         # EmbSize x SentSize
        positive_sims = torch.bmm(batch_embs, positive_embs)   
        positive_probs = torch.sigmoid(torch.bmm(batch_embs, positive_embs)) # SentSize x SentSize
        positive_loss = torch.nn.functional.binary_cross_entropy(positive_probs * self.positive_sim_mask,
                                               self.positive_sim_mask.expand_as(positive_probs))

        # NegativeSampling
        negative_words = torch.randint(1, self.vocab_size,
                                       size=(batch_size, self.negative_samples_n))
        negative_embs = self.embeddings(negative_words).permute(0, 2, 1)     # EmbSize x NegSamplesN        
        negative_probs = torch.sigmoid(torch.bmm(batch_embs, negative_embs)) # SentSize x NegSamplesN
        negative_loss = torch.nn.functional.binary_cross_entropy(negative_probs, negative_probs.new_zeros(negative_probs.shape))

        return positive_loss + negative_loss # -> min

# Тренируем нейросеть

def Train(model, train_dataset, test_dataset,
                    learningRate=1e-2, epoch_n=3, batch_size=10,
                    max_batches_train=100000, max_batches_test=100000):

    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    best_loss = float('inf')
    best_model = copy.deepcopy(model)

    for epoch in range(epoch_n):
        model.train()
        batches_train = 0
        
        for (batch, _) in train_dataloader:
            if batches_train > max_batches_train:
                break

            pred = model(batch)
            model.zero_grad()
            pred.backward()
            optimizer.step()
            
            batches_train += 1

        model.eval()
        test_loss = 0
        batches_test = 0

        torch.no_grad()
        for (batch, _) in test_dataloader:
            if batches_test > max_batches_test:
                break
                
            pred = model(batch)
            test_loss += float(pred)
            batches_test += 1

        test_loss /= batches_test

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model)
        else:
            break
             
    return best_loss, best_model


# Инкапсуляция word2vec

class Word2Vec:
    def __init__(self, words, n=100, r=5, ns_n=10):
        train_words = words
        test_words = words

        vocabulary = words2vocabulary(words, padding_word='NULL')

        train_ids = words2ids(train_words, vocabulary)
        test_ids = words2ids(test_words, vocabulary)
        sentence_len = 20
        
        train_dataset = WordsDataset(train_ids,
                                        np.zeros(len(train_ids)),
                                        out_len=sentence_len)
        test_dataset = train_dataset
        
        nn = NN(len(vocabulary), n, sentence_len,
                        radius=r, negative_samples_n=ns_n)
        
        best_loss, best_model = Train(nn,
                                        train_dataset,
                                        test_dataset,
                                        learningRate=0.01,
                                        epoch_n=3,
                                        batch_size=10)                                        

        embs = best_model.embeddings.weight.detach().numpy()

        self.embeddings = embs / (np.linalg.norm(embs, ord=2, axis=-1, keepdims=True) + 1e-4)
        self.vocabulary = vocabulary
        self.id2word = {i: w for w, i in vocabulary.items()}

    def most_similar(self, word, topk=10):
        return self.most_similar_vector(self.get_vector(word), topk=topk)

    def most_similar_vector(self, query_vector, topk=10):
        if query_vector is None:
            return []

        topk = topk+1
        similarities = self.embeddings @ query_vector
        best_indices = np.argpartition(-similarities, topk, axis=0)[:topk]
        result = [(self.id2word[i], similarities[i]) for i in best_indices]
        result.sort(key=lambda x: -x[1])
        return result[1:]

    def get_vector(self, word):
        if word not in self.vocabulary:
            return None
        return self.embeddings[self.vocabulary[word]]


data = open('war_and_peace.txt', encoding='utf-8').read()

words = text2words(data, min_token_size=2)
sentences = [[words[i] for i in range(item - 20, item)] for item in range(0, len(words))]

start_time = time.time()
word2vec = Word2Vec(words)
# word2vec = gensim.models.Word2Vec(sentences=sentences,
                                #   window=5)
print("--- %s seconds ---" % (time.time() - start_time))


# print(word2vec.wv.most_similar('hussars', topn=10))
print(word2vec.most_similar("dragoons"))
print(word2vec.most_similar("hussars"))
print(word2vec.most_similar_vector(
    word2vec.get_vector("andrew") 
    - word2vec.get_vector("prince") 
    + word2vec.get_vector("princess")
))
