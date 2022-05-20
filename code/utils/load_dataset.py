import numpy as np
import os, pickle, random, torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models import Word2Vec
    
    
def seq_process(seq, max_len):
    """ padding and truncate sequence """
    if len(seq)<max_len:
        return seq+'*'*(max_len-len(seq))
    else:
        return seq[:max_len]

    
def seqItem2id(item, seq_type):
    """ Convert seq item to token """
    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'protein' else 'ATCG'
    seqItem2id = {}
    seqItem2id.update(dict(zip(items, range(1, len(items)+1))))
    seqItem2id.update({"*":0})
    return seqItem2id[item]


def id2seqItem(i, seq_type):
    """ Convert token to seq item  """ 
    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'protein' else 'ATCG'
    id2seqItem = ["*"]+list(items)
    return id2seqItem[i]


def vectorize(emb_type, seq_type, window=13, sg=1, workers=8):
    """ Get embedding of 'onehot' or 'word2vec-[dim] """
    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'protein' else 'ATCG'
    emb_path = os.path.join(r'../embeds/', seq_type)
    emb_file = os.path.join(emb_path, emb_type+'.pkl')
    
    if os.path.exists(emb_file):
        with open(emb_file, 'rb') as f:
            embedding = pickle.load(f)
        # print(f'Loaded cache from {emb_file}.')
        return embedding
    
    if emb_type == 'onehot':
        embedding = np.concatenate(([np.zeros(len(items))], np.eye(len(items)))).astype('float32')
        
    elif emb_type[:8] == 'word2vec':
        _, emb_dim = emb_type.split('-')[0], int(emb_type.split('-')[1])
        seq_data = pickle.load(open(r'../data/seq_data.pkl','rb'))[seq_type]
        doc = [list(i) for i in list(seq_data)]
        model = Word2Vec(doc, min_count=1, window=window, size=emb_dim, workers=workers, sg=sg, iter=10)
        char2vec = np.zeros((len(items) + 2, emb_dim))
        for i in range(len(items) + 2):
            if id2seqItem(i, seq_type) in model.wv:
                char2vec[i] = model.wv[id2seqItem(i, seq_type)]
        embedding = char2vec
    
    if os.path.exists(emb_path) == False:
        os.makedirs(emb_path)
    with open(emb_file, 'wb') as f:
        pickle.dump(embedding, f, protocol=4)
    # print(f'Loaded cache from {emb_file}.')
    return embedding


class CelllineDataset(Dataset):
    def __init__(self, indexes, seqs, labels, ccds_ids, emb_type, seq_type, max_len):
        self.indexes = indexes
        self.labels = labels
        self.num_ess = np.sum(self.labels == 1)
        self.num_non = np.sum(self.labels == 0)
        self.ccds = ccds_ids
        self.raw_seqs = seqs
        self.processed_seqs = [seq_process(seq, max_len) for seq in self.raw_seqs]
        self.tokenized_seqs = [[seqItem2id(i, seq_type) for i in seq] for seq in self.processed_seqs]
        embedding = nn.Embedding.from_pretrained(torch.tensor(vectorize(emb_type, seq_type)))
        self.emb_dim = embedding.embedding_dim
        self.features = embedding(torch.LongTensor(self.tokenized_seqs))
        
    def __getitem__(self, item):
        return self.features[item], self.labels[item]
    
    def __len__(self):
        return len(self.indexes)


def load_dataset(seq_type, emb_type, cell_line, max_len, seed):
    """ Load train & test dataset """
    seq_data = pickle.load(open('../data/seq_data.pkl','rb'))[seq_type]
    label_data = pickle.load(open('../data/label_data.pkl','rb'))[cell_line]
    ccds_ids = pickle.load(open('../data/gene_ann_data.pkl','rb'))['ccds_id']

    ess_indexes = [i for i, e in enumerate(label_data) if int(e) == 1]
    non_indexes = [i for i, e in enumerate(label_data) if int(e) == 0]
    num_ess = len(ess_indexes)
    num_non = len(non_indexes)
    # print(f'{cell_line} dataset   essential:{num_ess}   non-essential:{num_non}')  
    
     # split data with balanced test set
    random.seed(seed)
    random.shuffle(ess_indexes)
    random.shuffle(non_indexes)
    test_indexes = ess_indexes[:int(num_ess * 0.2)] + non_indexes[:int(num_ess * 0.2)]
    train_indexes = list(set(ess_indexes + non_indexes) - set(test_indexes))
    
    train_seqs = [seq_data[i] for i in train_indexes]
    train_labels = np.array([label_data[i] for i in train_indexes])
    train_ccds = [ccds_ids[i] for i in train_indexes]
    
    test_seqs = [seq_data[i] for i in test_indexes]
    test_labels = np.array([label_data[i] for i in test_indexes])
    test_ccds = [ccds_ids[i] for i in test_indexes]
      
    train_dataset = CelllineDataset(train_indexes, train_seqs, train_labels, train_ccds, emb_type, seq_type, max_len)
    test_dataset = CelllineDataset(test_indexes, test_seqs, test_labels, test_ccds, emb_type, seq_type, max_len)
    return train_dataset, test_dataset
    
