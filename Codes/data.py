from common import *

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
          '*', '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
          '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
          '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
          '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
          '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

# create dataset class
class QuoraDataset(Dataset):
    def __init__(self,mode="train"):
        self.mode = mode
        self.embed_size = 300  # how big is each word vector
        self.max_features = 120000  # how many unique words to use (i.e num rows in embedding vector)
        self.maxlen = 72  # max number of words in a question to use
        self.embedding_matrix = self.load_embedding_matrix()

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y  = np.eye(config.num_classes,dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)

        X = T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])])(X)
        # X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        return X.float(),y

    def load_embedding_matrix(self):
        tqdm.pandas()
        start_time = time.time()

        train_X, test_X, train_y, word_index = self.load_and_prec()
        embedding_matrix_1 = self.load_glove(word_index)
        total_time = (time.time() - start_time) / 60
        print("Took {:.2f} minutes".format(total_time))

        # embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis=0)
        embedding_matrix = embedding_matrix_1

        # embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis=1)
        print(np.shape(embedding_matrix))

        # del embedding_matrix_1, embedding_matrix_2
        del embedding_matrix_1

        return embedding_matrix

    def _get_mispell(self, mispell_dict):
        mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
        return mispell_dict, mispell_re

    def replace_typical_misspell(self, text):
        mispellings, mispellings_re = self._get_mispell(mispell_dict)
        def replace(match):
            return mispellings[match.group(0)]

        return mispellings_re.sub(replace, text)

    def load_and_prec(self):
        train_df = pd.read_csv(config.train_data)
        print("Train shape : ", train_df.shape)
        test_df = pd.read_csv(config.test_data)
        print("Test shape : ", test_df.shape)

        # lower
        train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())
        test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())

        # Clean the text
        train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: self.clean_text(x))
        test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: self.clean_text(x))

        # Clean numbers
        train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: self.clean_numbers(x))
        test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: self.clean_numbers(x))

        # Clean speelings
        train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: self.replace_typical_misspell(x))
        test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: self.replace_typical_misspell(x))

        ## fill up the missing values
        train_X = train_df["question_text"].fillna("_##_").values
        test_X = test_df["question_text"].fillna("_##_").values

        ## Tokenize the sentences
        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(list(train_X))
        train_X = tokenizer.texts_to_sequences(train_X)
        test_X = tokenizer.texts_to_sequences(test_X)

        ## Pad the sentences
        train_X = pad_sequences(train_X, maxlen=self.maxlen)
        test_X = pad_sequences(test_X, maxlen=self.maxlen)

        ## Get the target values
        train_y = train_df['target'].values

        # # shuffling the data
        # np.random.seed(SEED)
        # trn_idx = np.random.permutation(len(train_X))
        #
        # train_X = train_X[trn_idx]
        # train_y = train_y[trn_idx]

        return train_X, test_X, train_y, tokenizer.word_index

    def load_glove(self, word_index):
        EMBEDDING_FILE = os.path.join(config.embedding_dir, "glove.840B.300d", "glove.840B.300d.txt")
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        # word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def load_para(self, word_index):
        EMBEDDING_FILE = os.path.join(config.embedding_dir, "paragram_300_sl999", "paragram_300_sl999.txt")
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        # word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def clean_text(self, x):
        x = str(x)
        for punct in puncts:
            x = x.replace(punct, f' {punct} ')
        return x

    def clean_numbers(self, x):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def augumentor(self,image):

        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes(
                iaa.OneOf([
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Affine(shear=(-16, 16)),
                ])
            )
        ], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug



if __name__ == '__main__':
    train_data = QuoraDataset(mode="train")
