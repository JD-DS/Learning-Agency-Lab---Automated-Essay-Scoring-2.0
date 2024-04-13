
def load_embed(file, wiki_news_path):
    """
    Load the embeddings from a file.
    """
    print(f"Loading embeddings from {file}")
    
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == wiki_news_path:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(file), "Reading Embedding File") if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(file, encoding='latin'), "Reading Embedding File"))
    
    print(f"Loaded embeddings from {file}")
    return embeddings_index

def parallel_load_embeddings(paths):
    """
    Load multiple embeddings in parallel using multiprocessing.
    
    :param paths: List of paths to the embedding files.
    :return: Dictionary of embeddings.
    """
    with Pool(processes=len(paths)) as pool:
        embeddings = pool.starmap(load_embed, [(path, paths[-1]) for path in paths])
    return dict(zip(["glove", "paragram", "fasttext"], embeddings))

def embedding_checks(df, glove_path, paragram_path, wiki_news_path, col_name='clean_text'):
    """
    Load embeddings, build vocabulary from the DataFrame, and check the vocabulary coverage in the embeddings.
    
    :param df: DataFrame containing the text data.
    :param glove_path: Path to the GloVe embedding file.
    :param paragram_path: Path to the Paragram embedding file.
    :param wiki_news_path: Path to the Wiki News embedding file.
    :param col_name: Column name of the DataFrame to analyze.
    :return: Tuple of DataFrame, OOV words for GloVe, Paragram, and Wiki News embeddings.
    """
    paths = [glove_path, paragram_path, wiki_news_path]
    embeddings = parallel_load_embeddings(paths)

    embed_glove, embed_paragram, embed_fasttext = embeddings.values()
    
    def build_vocab(texts):
        """
        Build a vocabulary from a given list of texts.
        """
        print("Building vocabulary.")
        sentences = texts.apply(lambda x: x.split()).values
        vocab = {}
        for sentence in tqdm(sentences, desc="Populating Vocabulary"):
            for word in sentence:
                vocab[word] = vocab.get(word, 0) + 1
        print("Vocabulary built.")
        return vocab

    def check_coverage(vocab, embeddings_index):
        """
        Check which words in the vocabulary are covered by the embeddings.
        """
        print("Checking coverage.")
        known_words = {}
        unknown_words = {}
        for word in tqdm(vocab.keys(), desc="Checking Words"):
            if word in embeddings_index:
                known_words[word] = vocab[word]
            else:
                unknown_words[word] = vocab[word]
        print("Coverage checked.")
        return sorted(unknown_words.items(), key=lambda x: x[1], reverse=True)

    print("Processing dataset.")
    vocab = build_vocab(df[col_name])
    
    oov_glove = check_coverage(vocab, embed_glove)
    oov_paragram = check_coverage(vocab, embed_paragram)
    oov_fasttext = check_coverage(vocab, embed_fasttext)
  
    print("Processed dataset.")
    
    return df, oov_glove, oov_paragram, oov_fasttext


#############################################################################################


def clean_text(df, col_name = 'full_text'):
    """
    Preprocesses text for both training and testing datasets. 
    Includes loading embeddings, building vocabularies, cleaning text among other things.
    
    :param summaries_train: DataFrame with the training data
    :param summaries_test: DataFrame with the testing data
    :param glove_path: path to the GloVe embedding
    :param paragram_path: path to the Paragram embedding
    :param wiki_news_path: path to the Wiki News embedding
    
    :return: Preprocessed DataFrame and list of out-of-vocab words
    """
    print("Starting text cleaning process. \n")


    
    # Lowercase all texts

    df['lowered'] = df[col_name].apply(lambda x: x.lower())

    
    
    def clean_spacing(text):
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)

        # Ensure there is one space after punctuation
        text = re.sub(r'([,.!?])([^\s])', r'\1 \2', text)

        return text
    

    df['clean_text'] = df['lowered'].apply(lambda x: clean_spacing(x))


        
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                     "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                     'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])
        for p in punct:
            text = text.replace(p, f' {p} ')
        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
        for s in specials:
            text = text.replace(s, specials[s])
        return text
    

    df['clean_text'] = df['clean_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

    df['clean_text'] = df['lowered'].apply(lambda x: clean_spacing(x))


    cont_map = {
        "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",  "could've": "could have",
        "couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not",
        "hadn't've": "had not have","hasn't": "has not",
        "haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is",
        "how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have","I'll": "I will",
        "I'll've": "I will have","I'm": "I am","I've": "I have",
        "isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us",
        "ma'am": "madam","mayn't": "may not",
        "might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
        "mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
        "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
        "shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is",
        "should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would",
        "that'd've": "that would have","that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is",
        "they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are",
        "they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had",
        "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
        "weren't": "were not","what'll": "what will","what'll've": "what will have",
        "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
        "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is",
        "who've": "who have","why's": "why is",
        "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
        "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would",
        "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have",
        "you'll": "you you will","you'll've": "you you will have","you're": "you are",  "you've": "you have"}

    c_re = re.compile('(%s)' % '|'.join(cont_map.keys()))

    def expandContractions(text, c_re=c_re):
        def replace(match):
            return cont_map[match.group(0)]
        return c_re.sub(replace, text)


    p = inflect.engine()

    def removeHTML(text):
        """
        Remove HTML tags from a given text string using regex.
    
        Args:
            text (str): The input text string containing HTML tags.
    
        Returns:
            str: The text string with HTML tags removed.
        """
        html = re.compile(r'<.*?>')
        return html.sub('', text)

    def dataPreprocessing(text):
        """
        Process the input text to perform a series of cleaning and formatting tasks,
        including converting numbers to words, removing specific patterns and whitespace,
        and stripping unwanted characters.
    
        Args:
            text (str): The input text string to preprocess.
    
        Returns:
            str: The cleaned and formatted text.
        """
        text = text.lower()
        text = removeHTML(text)
        text = re.sub(r'\d+', lambda match: p.number_to_words(match.group()) + " ", text)  # Add spaces around the number words
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces to a single space
        text = re.sub("@\w+", '', text)
        text = re.sub("'\d+", '', text)
        text = re.sub("\d+", '', text)
        text = re.sub("http\w+", '', text)
        text = re.sub(r"[-_]+", " ", text)  # Replace hyphens and underscores with space
        text = expandContractions(text)
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"[^\w\s]", "", text)  # Remove all non-alphanumeric and non-space characters
        text = re.sub(r"\s+", " ", text)    # Normalize multiple spaces to a single space

        text = text.strip()

        return text

    
    df['clean_text'] = df['clean_text'].apply(lambda x: dataPreprocessing(x))

   

    # #punctuation removal

    # def remove_punct(text):
    #     table=str.maketrans('','',string.punctuation)
    #     return text.translate(table)
    
    # df['no_punct'] = df['clean_text'].apply(lambda x: remove_punct(x))

    # #stopwords removal

    # stop_words = set(stopwords.words('english'))

    # def remove_stopwords(text):
        
    #     word_tokens = word_tokenize(text)
    #     sent_tokens = sent_tokenize(text)

    #     filtered_words = [word for word in word_tokens if word.lower() not in stop_words]

    #     filtered_sent = [word for word in sent_tokens if word.lower() not in stop_words]

    #     words = ' '.join(filtered_words)

    #     sents = ' '.join(filtered_sent)

    #     return words, sents
    
    # df['word_tokens'], df['sent_tokens'] = zip(*df['clean_text'].map(remove_stopwords))

    return df



##########################################################################################################


def correct_spellings_batch(misspelled_words_batch):
    """
    Corrects the spelling of words in a batch.

    :param misspelled_words_batch: A batch of misspelled words to be corrected.
    :return: A list of tuples where each tuple contains the original word, the corrected word
             (or None if no correction was found), and a boolean indicating whether the word was corrected.
    """
    spell_checker = SpellChecker()

    results = []
    for word in misspelled_words_batch:
        corrected = spell_checker.correction(word)
        is_corrected = corrected != word and corrected is not None
        result = (word, corrected if is_corrected else None, is_corrected)
        results.append(result)

    return results

def main(misspelled_words):
    num_batches = multiprocessing.cpu_count()

    words_per_batch = len(misspelled_words) // num_batches
    batches = [misspelled_words[i:i + words_per_batch] for i in range(0, len(misspelled_words), words_per_batch)]

    results = process_map(correct_spellings_batch, batches, max_workers=num_batches)

    # Filter to include only corrected words and exclude where corrected is None
    corrected_words = [(original, corrected) for sublist in results for original, corrected, is_corrected in sublist if is_corrected and corrected is not None]
    uncorrected_words = [original for sublist in results for original, corrected, is_corrected in sublist if not is_corrected or corrected is None]

    return corrected_words, uncorrected_words


#############################################################################################

def apply_corrections_to_text(text, corrections):
    """
    Applies spelling corrections to the text.

    :param text: The original text to be corrected.
    :param corrections: A dictionary of original to corrected word mappings.
    :return: The text with applied spelling corrections.
    """
    words = text.split()  # Tokenize the text into words
    corrected_text = ' '.join([corrections.get(word, word) for word in words])
    return corrected_text

#############################################################################################


def save_list(data, filename):
    """
    Serialize a list and save it to a text file in JSON format.
    
    Parameters:
    - data (list): The list to be serialized and saved.
    - filename (str): The path to the file where the list will be saved.
    """
    with open(filename, 'w') as file:
        json.dump(data, file)


#############################################################################################




def custom_train_validation_split(essays, test_size=0.2, random_state=56):
    
    """
    Custom function to perform train-validation split ensuring that
    the same prompt IDs are in both training and validation sets.

    Parameters:
    - summaries: DataFrame containing summaries and associated prompt_ids
    - prompts: DataFrame containing prompts and associated prompt_ids
    - test_size: Proportion of the dataset to be used as the validation set
    - random_state: Random seed for reproducibility

    Returns:
    - train_summaries: Training set containing summaries
    - validation_summaries: Validation set containing summaries
    - train_prompts: Training set containing prompts
    - validation_prompts: Validation set containing prompts
    """
    
    # Extract unique prompt IDs
    unique_essay_ids = essays['essay_id'].unique()

    # Split the unique prompt IDs into training and validation sets
    train_ids, validation_ids = train_test_split(unique_essay_ids, test_size=test_size, random_state=random_state)

    # Use these IDs to filter the original summaries and prompts DataFrames
    train_essays = essays[essays['essay_id'].isin(train_ids)]
    validation_essays = essays[essays['essay_id'].isin(validation_ids)]

    return train_essays, validation_essays

#############################################################################################

def extract_features(essay, col='corrected_text', tfidf_vectorizer=None):
    """
    Extracts TF-IDF features from a column of texts in a DataFrame.
    
    Parameters:
    - essay (DataFrame): DataFrame containing the essays.
    - col (str): The column name of the text data.
    - tfidf_vectorizer (TfidfVectorizer, optional): A pre-fitted TfidfVectorizer. If None, a new one will be fitted.
    
    Returns:
    - DataFrame: The DataFrame with TF-IDF features merged.
    - TfidfVectorizer: The fitted or provided TfidfVectorizer instance.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer='word',
            ngram_range=(1,5),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
        )
        train_tfid = tfidf_vectorizer.fit_transform(essay[col])
    else:
        train_tfid = tfidf_vectorizer.transform(essay[col])
    
    dense_matrix = train_tfid.toarray()
    df = pd.DataFrame(dense_matrix, columns=[f'tfid_{i}' for i in range(train_tfid.shape[1])])
    df['essay_id'] = essay['essay_id']
    train_feats = pd.merge(essay, df, on='essay_id', how='left')

    return train_feats, tfidf_vectorizer

# essays, tfidf_vectorizer = extract_features(essays, col='corrected_text',
    #                                                      tfidf_vectorizer=tfidf_vectorizer)


def preprocess_data(essays, tfidf_vectorizer=None):
    
    """
    Preprocesses and computes features for a dataset with text summaries and prompts.
    
    Parameters:
    - summaries (DataFrame): DataFrame containing the text summaries.
    - prompts (DataFrame): DataFrame containing the text prompts.
    - tfidf_vectorizer (TfidfVectorizer, optional): A pre-fitted TfidfVectorizer.
    
    Returns:
    - DataFrame: The preprocessed and feature-engineered DataFrame.
    """
    
    print('Preprocessing Data.......')

    # Extract TF-IDF features

    # essays, tfidf_vectorizer = extract_features(essays, col='corrected_text',
    #                                                      tfidf_vectorizer=tfidf_vectorizer)

    stop_words = set(stopwords.words('english'))
    
    print('Generating Text Based Features.......')

    # Compute word count, sentence count, text length, and stopword count
    essays['word_count'] = essays['corrected_text'].apply(lambda x: len(word_tokenize(x)))
    essays['sentence_count'] = essays['corrected_text'].apply(lambda x: len(sent_tokenize(x)))
    essays['len_text'] = essays['corrected_text'].str.len()
    essays['stop_count'] = essays['corrected_text'].apply(lambda x: len([word for word in word_tokenize(x) if word in stop_words]))
    
    import string

    # essays['punct_count'] = essays['corrected_text'].apply(lambda x: len([char for char in x if char in string.punctuation]))
    # essays['capital_count'] = essays['corrected_text'].apply(lambda x: len([word for word in word_tokenize(x) if word.isupper()]))
    
    from nltk import pos_tag

    essays['noun_count'] = essays['corrected_text'].apply(lambda x: len([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('NN')]))
    
    from nltk import ne_chunk

    essays['ne_count'] = essays['corrected_text'].apply(lambda x: len([chunk for chunk in ne_chunk(pos_tag(word_tokenize(x))) if hasattr(chunk, 'label')]))
    essays['avg_word_len'] = essays['corrected_text'].apply(lambda x: sum(len(word) for word in word_tokenize(x)) / len(word_tokenize(x)) if len(word_tokenize(x)) > 0 else 0)
    essays['lex_div'] = essays['corrected_text'].apply(lambda x: len(set(word_tokenize(x))) / len(word_tokenize(x)) if len(word_tokenize(x)) > 0 else 0)
    
    from textblob import TextBlob

    essays['polarity'] = essays['corrected_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    essays['subjectivity'] = essays['corrected_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    essays['flesch_score'] = essays['corrected_text'].apply(lambda x: flesch_reading_ease(x))

    
    from collections import Counter

    # essays['most_common_word_count'] = essays['corrected_text'].apply(lambda x: Counter(word_tokenize(x)).most_common(1)[0][1] if len(word_tokenize(x)) > 0 else 0)

    
    # # Calculate cosine similarity between the text and its corresponding prompt
    # print('Computing Cosine Similarity.......')
    # merged_data = compute_cosine_similarity(merged_data, 
    #                                         'treated_question_summaries', 
    #                                         'treated_question_prompts')
    
    return essays, tfidf_vectorizer


def compute_cosine_similarity(df, text_col, content_col):
    
    """
    Computes the cosine similarity between two columns of text in a DataFrame.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the texts.
    - text_col (str): The name of the column containing the first set of texts.
    - content_col (str): The name of the column containing the second set of texts.
    
    Returns:
    - DataFrame: The DataFrame with an additional column for the computed cosine similarity.
    """
    
    # Combine texts from both columns to fit the TF-IDF vectorizer
    all_texts = df[text_col].tolist() + df[content_col].tolist()
    
    # Fit the TF-IDF vectorizer on the combined corpus
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    
    # Generate TF-IDF vectors for both columns
    text_tfidf = vectorizer.transform(df[text_col])
    content_tfidf = vectorizer.transform(df[content_col])
    
    # Compute cosine similarity for each pair of text and content
    cosine_sim_values = [cosine_similarity(text_tfidf[i], content_tfidf[i])[0][0] for i in range(len(df))]
    
    # Add the computed cosine similarity values to the DataFrame
    df['cos_sim'] = cosine_sim_values
    
    return df


#############################################################################################



def bert_spell_check_df(df, column_name, misspelled_words):
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fill_mask = pipeline('fill-mask', model=model_name, top_k=1)  # Setting top_k for speed
    max_length = tokenizer.model_max_length  # Get the maximum length the model can handle

    corrections = {}

    def escape_regex_special_chars(text):
        """ Escape regex special characters in a given text """
        return re.escape(text)

    def chunk_text(text, size):
        """ Split text into chunks where each chunk has a maximum number of tokens `size` """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(tokenizer.tokenize(word)) <= size:
                current_chunk.append(word)
                current_length += len(tokenizer.tokenize(word))
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(tokenizer.tokenize(word))
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def spell_check_text(text):
        """ Correct misspelled words in a given text based on predefined list using BERT """
        corrected_text = text
        chunks = chunk_text(corrected_text, max_length - 50)  # Reduce max length a bit for safety margin
        corrected_chunks = []

        for chunk in chunks:
            for word in misspelled_words:
                pattern = rf'\b{escape_regex_special_chars(word)}\b'
                if re.search(pattern, chunk):
                    sentences = re.split(r'(\.|\?|!)\s+', chunk)
                    for i, sentence in enumerate(sentences):
                        if word in sentence:
                            tokens = tokenizer.tokenize(sentence)
                            if len(tokens) > max_length:
                                print(f"Sentence too long for BERT processing: {sentence}")
                                continue

                            masked_sentence = re.sub(pattern, tokenizer.mask_token, sentence, count=1)
                            predictions = fill_mask(masked_sentence)
                            if predictions:
                                best_prediction = predictions[0]['sequence']
                                split_prediction = best_prediction.split(tokenizer.mask_token)
                                if len(split_prediction) > 1:
                                    corrected_piece = split_prediction[1].strip()
                                    corrections[word] = corrected_piece
                                    sentences[i] = re.sub(pattern, corrected_piece, sentence, count=1)
                            else:
                                print(f"No prediction for {word} in sentence: {sentence}")
                    chunk = ''.join(sentences)
            corrected_chunks.append(chunk)

        return ' '.join(corrected_chunks)

    # Add tqdm progress bar for DataFrame processing
    
    tqdm.pandas(desc="Processing DataFrame Rows")
    df[f'{column_name}_corrected'] = df[column_name].progress_apply(spell_check_text)

    return df, corrections


#############################################################################################






