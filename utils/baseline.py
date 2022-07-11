from spacy.matcher import Matcher
# from spacy.matcher import PhraseMatcher
from os.path import join
import ast
import json



def run_baseline_en(nlp, doc, t_matcher):
    
    appreciation = False
    patterns = []
    sequences = []
    
    praise = ["you're the best"]
    negated = ["don't thank you", "please? thank", "please thank", "fuck", "fucked", "ruined", "ruining", "wasting"]
    request = ["could you", "can you", "would you", "please?" , "would be nice if", "would be great if", "would it be possible", "is it possible"]
    
    t_matches = t_matcher(doc)
    
    request_match = []
    if t_matches:
        appreciation = True
        for match_id, start, end in t_matches:
            span = doc[start:end]
            # discard the match if captured sequence is actually a request-Gratitude (e.g. Can you email me the file please? Thanks)
            first_part = doc[:start].text.lower()
            request_match = [x for x in request if x in first_part]
            patterns.append(nlp.vocab.strings[match_id])
            sequences.append(span.text)
    
    for text in praise:
      if text in doc.text.lower():
        appreciation = True
    
    
    for text in negated:
      if text in doc.text.lower():
        appreciation = False
    
    if request_match:
      appreciation = False
    
    
    if appreciation:
      return appreciation, patterns, sequences
    else:
      return appreciation, [], []

def run_baseline_fr(nlp, doc, t_matcher):
    
    appreciation = False
    patterns = []
    sequences = []
    
    praise = []
    negated = ["me remercie", "seigneur", "merci d'avance"]
    request = ["pouvez-vous", "pourriez-vous", "peux-tu", "est-il possible" , "serait-il possible"]
    
    t_matches = t_matcher(doc)
    
    request_match = []
    if t_matches:
        appreciation = True
        for match_id, start, end in t_matches:
            span = doc[start:end]
            # discard the match if captured sequence is actually a request-Gratitude (e.g. Can you email me the file please? Thanks)
            first_part = doc[:start].text.lower()
            request_match = [x for x in request if x in first_part]
            patterns.append(nlp.vocab.strings[match_id])
            sequences.append(span.text)
    
    for text in praise:
      if text in doc.text.lower():
        appreciation = True
    
    
    for text in negated:
      if text in doc.text.lower():
        appreciation = False
    
    if request_match:
      appreciation = False
    
    
    if appreciation:
      return appreciation, patterns, sequences
    else:
      return appreciation, [], []

def load_t_matcher(PATH_CONFIG, config_file, nlp):
  """
  load patterns and matcher
  """
  with open(join(PATH_CONFIG, config_file), 'r') as f:
    appreciation_patterns = ast.literal_eval(str(json.load(f)))
  matcher = Matcher(nlp.vocab)
  [matcher.add(rule, [pattern]) for rule, pattern in appreciation_patterns.items()] 
  
  # #add emoji patterns
  # grat_emojis = ["🙏", "🙏🏻", "🙏🏼", "🙏🏼", "🙏🏽", "🙏🏿", "🙏🏾"]
  # grat_emoji_patterns = [[{"ORTH": emoji}] for emoji in grat_emojis]
  # matcher.add("GRATEFUL_EMOJIS", grat_emoji_patterns)
  
  return matcher



# def load_p_matcher(PATH_CONFIG, config_file, nlp):
#   """
#   load terminology and matcher
#   """
#   with open(join(PATH_CONFIG, config_file), 'r', encoding="utf8") as f:
#     dictionary = ast.literal_eval(str(json.load(f)))
    
#   terminology = dictionary["Appreciation Terminology"]
#   matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
#   res = []
#   [res.append(x) for x in terminology if x not in res]
#   patterns = [nlp.make_doc(text) for text in res]
#   matcher.add("Appreciation Aexicon", patterns)
#   return matcher
