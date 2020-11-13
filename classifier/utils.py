import torch, os, pickle, nltk, re
import torch.utils.data as data
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)
import pandas as pd
import numpy as np
from tqdm import tqdm

import pprint
pp = pprint.PrettyPrinter(indent=1)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

UNK_idx = 0
PAD_idx = 1
SOS_idx = 2

def get_vocab_from_bert():
    tokenizer.save_vocabulary('./vocab.txt')


def get_info_from_external_knowledge(external_knowledge='SenticNet'):

    if external_knowledge == 'SenticNet':
        if os.path.exists('./senticnet_info.pkl'):
            with open('./senticnet_info.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            embeddings = dict()
            primary_emotions = dict()
            second_emotions = dict()
            similar_words = dict()
            polarity = dict()
            from senticnet6 import senticnet
            SN = senticnet
            for key, value in SN.items():
                key = key.replace('_', ' ')
                # key = tokenizer.encode(key, add_special_tokens=False)
                embeddings[key] = [float(s) for s in value[:4]]
                primary_emotions[key] = value[4]
                second_emotions[key] = value[5]
                polarity[key] = float(value[7])
                similar_words[key] = [words.replace('_', ' ') for words in value[8:]]
            tmp = [embeddings, primary_emotions, second_emotions, similar_words, polarity]
            with open('./senticnet_info.pkl', 'wb') as f:
                pickle.dump(tmp, f)
        return tmp

    elif external_knowledge == 'ConceptNet':
        pass

    else:
        raise ValueError('external_knowledge must `SenticNet` or `ConceptNet`.')


class Dataset(data.Dataset):
    def __init__(self, data, glove=False):
        self.glove = glove
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19,
            'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25,
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        item = {}
        item['sequence_text'] = self.data['sequence'][index]
        item['label_text'] = self.data['label'][index]
        item['sequence'], item['mask'] = self.preprocess(self.data['sequence'][index])
        item['emotion'], item['label'] = self.preprocess_emo(self.data['label'][index])

        return item

    def preprocess(self, sequence):
        if self.glove:
            word2index = pickle.load(open('./data/vocab.txt', 'rb'))
            input_ids = [SOS_idx] + [word2index[w] if w in word2index.keys() else UNK_idx for w in sequence]
            mask = [1 for i in range(len(input_ids))]
            while len(input_ids) < 50:
                input_ids.append(PAD_idx)
                mask.append(0)
            if len(input_ids) > 50:
                input_ids = input_ids[:50]
                mask = mask[:50]

            return input_ids, mask

        else:
            sequence = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=50,
                                             padding='max_length', return_attention_mask=True, truncation=True)
            return sequence['input_ids'], sequence['attention_mask']

    def preprocess_emo(self, emotion, emo_map=None):
        if emo_map == None:
            emo_map = self.emo_map
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        # >>>>>>>>>> one hot mode and label mode >>>>>>>>>> #
        return program, emo_map[emotion]


class EKDataset(Dataset):
    def __init__(self, data, config):
        self.glove = config.glove
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19,
            'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25,
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
        self.info = get_info_from_external_knowledge()
        self.embeddings, self.primary_emotions, self.second_emotions, self.similar_words, self.polarity = self.info

        # self.wordlist = sorted(self.embeddings.keys(), key=lambda x: len(x.split()), reverse=True)
        self.wordlist = self.embeddings.keys()
        wordkeys4 = '|'.join([' '+re.escape(word)+' ' for word in self.wordlist if len(word.split()) == 4])
        wordkeys3 = '|'.join([' '+re.escape(word)+' ' for word in self.wordlist if len(word.split()) == 3])
        wordkeys2 = '|'.join([' '+re.escape(word)+' ' for word in self.wordlist if len(word.split()) == 2])
        wordkeys1 = '|'.join([' '+re.escape(word)+' ' for word in self.wordlist if len(word.split()) == 1])
        # print(wordkeys1)
        self.keys = [wordkeys4, wordkeys3, wordkeys2, wordkeys1]

    def __getitem__(self, index):
        item = {}
        item['sequence_text'] = self.data['sequence'][index]
        item['label_text'] = self.data['label'][index]
        item['sequence'], item['mask'], item['scores'], item['ex_emb'] = self.preprocess(self.data['sequence'][index])
        item['emotion'], item['label'] = self.preprocess_emo(self.data['label'][index])

        return item

    def preprocess(self, sequence):
        if self.glove:
            word2index = pickle.load(open('./data/vocab.txt', 'rb'))
            input_ids = [SOS_idx] + [word2index[w] if w in word2index.keys() else UNK_idx for w in sequence]
            mask = [1 for i in range(len(input_ids))]
            while len(input_ids) < 50:
                input_ids.append(PAD_idx)
                mask.append(0)
            if len(input_ids) > 50:
                input_ids = input_ids[:50]
                mask = mask[:50]

            wordLevel = [[0, 0, 0, 0, 0]]
            occurance = list()
            sequence = ' '.join([' '] + sequence + [' '])
            for i in range(4):
                res = re.findall(self.keys[i], sequence)
                if res != []:
                    for w in res:
                        word = w[1:-1].replace(' ', '_')
                        sequence = sequence.replace(w[1:-1], word)
                        occurance.append(word)

            pos, neg, positive, negative = [], [], [], []

            for word in sequence.split():
                if word in occurance:
                    w = word.replace('_', ' ')
                    for i in range(len(w)):
                        wordLevel.append(self.embeddings[w] + [self.polarity[w]])
                    if self.polarity[w] > 0:
                        pos.append(self.polarity[w])
                    else:
                        neg.append(self.polarity[w])
                else:
                    wordLevel.append([0 for i in range(5)])
            while len(wordLevel) < 50:
                wordLevel.append([0, 0, 0, 0, 0])
            if len(wordLevel) > 50:
                wordLevel = wordLevel[:50]

            pos = 0 if pos == [] else np.mean(pos)
            neg = 0 if neg == [] else np.mean(neg)

            return input_ids, mask, [pos, neg], wordLevel


        else:
            tokenizer_info = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=50,
                                    padding='max_length', return_attention_mask=True, truncation=True)

            occurance = list()
            sequence = ' '.join([' ']+sequence+[' '])
            for i in range(4):
                res = re.findall(self.keys[i], sequence)
                if res != []:
                    for w in res:
                        word = w[1:-1].replace(' ', '_')
                        sequence = sequence.replace(w[1:-1], word)
                        occurance.append(word)

            pos, neg, positive, negative = [], [], [], []
            for word in sequence.split():
                if word in occurance:
                    w = word.replace('_', ' ')
                    if self.polarity[w] > 0:
                        pos.append(self.polarity[w])
                        positive.append(self.embeddings[w])
                    else:
                        neg.append(self.polarity[w])
                        negative.append(self.embeddings[w])

            positive = np.mean(positive, axis=0).tolist() if pos != [] else [0, 0, 0, 0]
            negative = np.mean(negative, axis=0).tolist() if neg != [] else [0, 0, 0, 0]
            pos = np.mean(pos) if pos != 0 else 0
            neg = np.mean(neg) if neg != 0 else 0

            return tokenizer_info['input_ids'], tokenizer_info['attention_mask'], [pos, neg], [positive, negative]

def load_dataset(config):
    if config.glove == True:
        dataset_path = './data/dataset_preproc_glove.p'
    else:
        dataset_path = './data/dataset_preproc_bert.p'

    if os.path.exists(dataset_path):
        print('LOADING dataset')
        with open(dataset_path, 'rb') as f:
            [data_tra, data_val] = pickle.load(f)
    else:
        print('Building dataset...')
        data = pd.read_csv('./data/prompt.csv', sep='\t', encoding='UTF8', header=0)
        data = data.sample(frac=1).reset_index(drop=True)
        data_tra, data_val = process_raw(data, config)
        with open(dataset_path, "wb") as f:
            pickle.dump([data_tra, data_val], f)
            print("Saved PICKLE")

    return data_tra, data_val


def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


class Lang:
    """
    create a new word dictionary, including 3 dictionaries:
    1) word to index;
    2) word and its count;
    3) index to word;
    and one counter indicating the number of words.
    """

    def __init__(self, init_index2word):
        """
        :param init_index2word: a dictionary containing (id: token) pairs
        """
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def process_raw(dataframe, config):
    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}
    data_train = {'label': [], 'sequence': []}
    data_dev = {'label': [], 'sequence': []}
    total_len = len(dataframe)
    print('There are {} training examples'.format(int(total_len * 0.9)))
    vocab = Lang({UNK_idx: "UNK", PAD_idx: "PAD", SOS_idx: "SOS"})
    for idx, row in dataframe.iterrows():
        emotion, text = row
        if idx < total_len * 0.9:
            data_train['label'].append(emotion)
            words = clean(text, word_pairs)
            vocab.index_words(words)
            data_train['sequence'].append(words)
        else:
            data_dev['label'].append(emotion)
            words = clean(text, word_pairs)
            vocab.index_words(words)
            data_dev['sequence'].append(words)

    if config.glove:
        if not os.path.exists(config.emb_path) or not os.path.exists(config.vocab_path):
            gen_embeddings(vocab, config)

    return data_train, data_dev


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def gen_embeddings(vocab, config):
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    pre_trained = 0
    for line in tqdm(open(config.emb_file, 'r', encoding='UTF8').readlines()):
        word2index = vocab.word2index
        sp = line.split()
        if len(sp) == config.emb_dim + 1:
            if sp[0] in vocab.word2index:
                embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
                pre_trained += 1
    print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    pickle.dump(embeddings, open(config.emb_path, 'wb'))
    pickle.dump(word2index, open(config.vocab_path, 'wb'))


def collate_fn(data):
    def getlen(masks):
        lengths = [sum(mask) for mask in masks]
        return lengths

    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    lengths = torch.LongTensor(getlen(item_info['mask']))
    sequence = torch.LongTensor(item_info['sequence'])
    label = torch.LongTensor(item_info['label'])
    emotion = torch.LongTensor(item_info['emotion'])
    mask = torch.LongTensor(item_info['mask'])

    if torch.cuda.is_available():
        lengths = lengths.cuda()
        sequence = sequence.cuda()
        label = label.cuda()
        emotion = emotion.cuda()
        mask = mask.cuda()

    d = {}
    d['lengths'] = lengths
    d['sequence'] = sequence
    d['label'] = label
    d['emotion'] = emotion
    d['mask'] = mask

    return d


def ek_collate_fn(data):
    def getlen(masks):
        lengths = [sum(mask) for mask in masks]
        return lengths

    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    lengths = torch.LongTensor(getlen(item_info['mask']))
    sequence = torch.LongTensor(item_info['sequence'])
    label = torch.LongTensor(item_info['label'])
    emotion = torch.LongTensor(item_info['emotion'])
    mask = torch.LongTensor(item_info['mask'])
    scores = torch.FloatTensor(item_info['scores'])
    ex_emb = torch.FloatTensor(item_info['ex_emb'])

    if torch.cuda.is_available():
        lengths = lengths.cuda()
        sequence = sequence.cuda()
        label = label.cuda()
        emotion = emotion.cuda()
        mask = mask.cuda()
        scores = scores.cuda()
        ex_emb = ex_emb.cuda()

    d = {}
    d['lengths'] = lengths
    d['sequence'] = sequence
    d['label'] = label
    d['emotion'] = emotion
    d['mask'] = mask
    d['scores'] = scores
    d['ex_emb'] = ex_emb

    return d


def prepare_data_seq(config):
    batch_size = config.batch_size
    data_tra, data_val = load_dataset(config)

    if config.external_knowledge:
        dataset_train = EKDataset(data_tra, config)
        data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                      batch_size=batch_size,
                                      shuffle=True, collate_fn=ek_collate_fn)

        dataset_valid = EKDataset(data_val, config)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                      batch_size=batch_size,
                                      shuffle=True, collate_fn=ek_collate_fn)

    else:
        dataset_train = Dataset(data_tra, config)
        data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                      batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_fn)

        dataset_valid = Dataset(data_val, config)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                      batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_fn)

    return data_loader_tra, data_loader_val







