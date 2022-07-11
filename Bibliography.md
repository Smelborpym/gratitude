# Emotion
For a quick top survey on the subject consider the video [here](https://www.youtube.com/watch?v=mzw6zFZ29y4)

A variety of different theories of emotional state:

* Johnson-Laird and Oatley (1989) distinguish 5 basic emotions: anger, disgust, fear, happiness, sadness. All others are analysed as generic, relational, caused, causative, goal or complex.
* Ekman: anger, disgust, fear, happiness, sadness and surprise.
* multi-dimensional theories:
 * PAD: dominance sometimes is not universal across languages
		* Pleasure vs. displeasure (how pleasant?)
 		* Arousal vs. non-arousal (how intense)
 		* Dominance vs. submissiveness (e.g. anger vs. fear)
 * EPA: quite reliable across languages
 		* Evaluation: good/bad
 		* Potency: strong/weak
 		* Activity: active/passive
 * psychometric measures ‘calmness’, ‘vitality’, etc....
 * activation, valence, potency, emotion intensity
 * Emotion classification in Social Media can use "distant supervision": indirect proxies for annotations as done [here](https://aclanthology.org/E12-1049.pdf). Results are summarized below
![distant supervision](/Users/mehdi_mdaghri/Desktop/Intent Detection Fear/sld2.png)

Typically it is assumed that particular words and phrases are
associated with these categories. But emotion classification is difficult for humans as explained [here](http://web.eecs.umich.edu/~mihalcea/papers/strapparava.acm08.pdf) and shown below
![human classif](/Users/mehdi_mdaghri/Desktop/Intent Detection Fear/sld1.png)

There is an interesting [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7836781/) trying to classify sychological fear and anxiety caused by COVID-19: Insights from Twitter analytics

# Bibliography
Ressources on "Papers on Emotion Cause Analysis" on GitHub [here](https://github.com/stevehamwu/Emotion-Cause-Analysis-Papers)

 * Task & Dataset
 * Emotion Cause Extraction (ECE)
	* Rule-based Methods
	* Deep learning-based Methods as in:
	* Reranking
	* Memory Network
	* Joint Learning
	* Hierarchical Network
	* Attention
	* GCN
 * Emotion Cause Pair Extraction (ECPE)
   * Rank
   * Match/Link/Attention
   * Transition
   * Sequence Labeling
   * GCN 


# Datasets

## Datasets applied to Emotion detection

### One single dataset containing multiple sources and data:

* There is a good ressource**(USED)** [here](https://github.com/sarnthil/unify-emotion-datasets/tree/master/datasets) with link to a GitHub [repository](https://github.com/sarnthil/unify-emotion-datasets)


### Pure tweets datasets:

* SSEC 	The SSEC corpus is an annotation of the SemEval 2016 Twitter stance and sentiment corpus with emotion labels. [Link](http://www.romanklinger.de/ssec/)

* suggested by Arnault with interesting dataset that might be in english or bengali. Link to the article [here](https://arxiv.org/pdf/1907.07826.pdf)

## Final datasets to be used:
* Construct a test dataset
* Construct a development dataset

## Baseline model (3 steps)

* Baseline 1: Human classif via Lexicon. This [paper](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-014-0031-z#ref-CR78) does exactly that for the Boston bombings and have nice figures to reproduce. It considers the Sentisense Lexicon available for download [here](http://nlp.uned.es/~jcalbornoz/resources/sentisense/).

>The SentiSense Affective Lexicon is a concept-based affective lexicon. It is intended to be used in sentiment analysis-related tasks, especially in polarity and intensity classification and emotion identification. SentiSense attaches emotional meanings to concepts from the WordNet lexical database, instead of terms, thus allowing addressing the word ambiguity problem using one of the many WordNet-based word sense disambiguation algorithms. SentiSense is available in English and Spanish.

* Baseline 2: consider the following 2 articles on easy methodologies to follow to beat the baseline model.
How to use the following [paper](https://www.medrxiv.org/content/10.1101/2021.01.16.21249943v1.full) to derive a baseline model to beat.
> The two-fold objectives of this paper are: (a) to develop and AI framework based on machine learning models with for emotion detection and (b) to pilot this model on unstructured tweets that followed quarantine regulation using stay at home messaging during the first wave of COVID-19. We investigate emotions and semantic structures to discover knowledge from general public tweeter exchanges. We analyze the structure of vocabulary patterns used on Twitter with a specific focus on the impact of the Stay-At-Home public health order during the first wave of the COVID-19. 


