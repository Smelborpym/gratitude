import re
from itertools import groupby 


def preprocess(text):
    """
    Ojective: Preprocess single text document for Gratitude
      Inputs: 
        text, str: Text to preprocess
     Outputs:
        p_text, str: Preprocessed text 
    """
    p_text = re.sub('RT', '', text)
    p_text = re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})+', '', p_text, flags=re.MULTILINE)
    p_text = re.sub('#', '', p_text, flags=re.MULTILINE)
    p_text = re.sub('[\w.+-]+@[\w-]+\.[\w.-]+', '', p_text, flags=re.MULTILINE)
    p_text = re.sub('([@][\w\.0-9\'_-]+)', 'USER ', p_text, flags=re.MULTILINE)
    # p_text = re.sub('[ USER]+', 'USER ', p_text, flags=re.MULTILINE)
    p_text = re.sub('\s+', ' ', p_text, flags=re.MULTILINE)
    p_text = re.sub('\r|\n', ' ', p_text, flags=re.MULTILINE)
    return p_text
  

def additional_preprocessing(text):
    """
    Apply additional experimentational preprocessing to the text
    """
    #remove mention successions
    p_text = ' '.join([i[0] for i in groupby(text.split())])
    
    return p_text 