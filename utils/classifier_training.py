import spacy




def create_word2vec_dataset(data, nlp, is_train=True):
    
    """
    Objective: Create spacy word2vec dataset for classification
       Inputs:
          dataset, pd.DataFrame: Orginal text data
          nlp, spacy.Language: spaCy language model
          is_train, bool: Is the desired dataset for training?
       Outputs:
          X, Y, list: Lists of vectorized text examples and their labels 
    """
    
    texts = data['text'].values.tolist()
    docs = list(nlp.pipe(texts))
    X = [doc.vector for doc in docs]
    if is_train:
        Y = data['label'].values.tolist()
        return X, Y
    else:
        return X
    
    
    
def create_textcat_dataset(texts, labels):
    
    """
    Objective: format training data for spacy's textcat
       Inputs:
          dataset, pd.DataFrame: Orginal text data
          nlp, spacy.Language: spaCy language model
       Outputs:
          training_examples, list: List of (text, {"cats": {"gratitude":label}}) tuples 
    """
    formatted_data = []
    for text, label in zip(texts, labels):
        formatted_data.append((text, {"cats":{"gratitude":label}}))
    return formatted_data

def make_textcat_predictions(data, nlp):
    
    """
    Objective: predict gratitude using trained classifier
       Inputs:
          dataset, pd.DataFrame: Orginal testing text data
          nlp, spacy.Language: Customized spaCy language model
       Outputs:
          predictions, list: List of predicted gratitude scores
    """
    texts = data['text'].values.tolist()
    docs = list(nlp.pipe(texts))
    predictions = [doc.cats["gratitude"] for doc in docs]
    return predictions
    
    