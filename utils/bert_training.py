from os.path import join
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.utils import shuffle




def get_model(max_length, transformer_model, num_labels, name_model=False, PATH_MODELS=False):
  
  """
	Get a model from scratch or if we have weights load it to the model.
	
	Inputs:
		- max_length, int: the input shape of the data
		- transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
																					we will use as base
																					(embedding model - sentence here)
		- num_labels, int: the number of intents
		- name_model (optional), str: look for an already existing model should be the entire path
	Outputs:
		- model, tensorflow.python.keras.engine.functional.Functional: the final model we'll train 
	"""
  
  logging.info('Creating architecture...')
  input_ids_in = tf.keras.layers.Input(shape=(max_length,), name='input_token', dtype='int32')
  input_masks_in = tf.keras.layers.Input(shape=(max_length,), name='masked_token', dtype='int32')
  embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0][:,0,:]
  transf_out = tf.keras.layers.Flatten()(embedding_layer)
  
  ## YOU CAN ADD A DENSE LAYER HERE OF COURSE
  dense1 = tf.keras.layers.Dense(128)(transf_out)
  dense2 = tf.keras.layers.Dense(512, activation='relu')(dense1)
  dense3 = tf.keras.layers.Dense(128)(dense2)
  output = tf.keras.layers.Dense(num_labels, activation='sigmoid')(dense3)
  model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = output)
  if name_model:
    try:
      model.load_weights(join(PATH_MODELS, name_model + '.h5'))
      logging.info('Model {} restored'.format(name_model))
    except:
      logging.warning('Model {} not found'.format(name_model))
      logging.warning('If training: new model from scratch')
      logging.warning('If classifying: the configuration does not fit the architecture and this model is not trained yet!')
  return model

def get_batches(X_train, y_train, tokenizer, batch_size, max_length, balanced=True):
	"""
	Objective: from features and labels yield a random batch of batch_size of (features, labels),
			   each time we reached all data we shuffle again the (features, labels) 
			   and we do it again (infinite loop)
			   
	Inputs:
		- X_train, np.array: the texts (features)
		- y_train, np.array: the labels
		- tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
		- batch_size, int: the size of the batch we yield
		- max_length, int: the input shape of the data
	Outputs: (generator)
		- inputs, np.array : two arrays one with ids from the tokenizer, and the masks associated with the padding
		- targets, np.array: the label array of the associated inputs
	"""

	if balanced:
		x = 1
#		_X_train, _y_train = balance_dataset(X_train, y_train)
#	else:
	_X_train, _y_train = X_train, y_train

	_X_train, _y_train = shuffle(_X_train, _y_train, random_state=11)

	i, j = 0, 0

	while i > -1:

		if (len(_X_train) - j*batch_size) < batch_size:
			j = 0

			if balanced:
				x = 1
#				_X_train, _y_train = balance_dataset(X_train, y_train)
#			else:
			_X_train, _y_train = X_train, y_train
      
			_X_train, _y_train = shuffle(_X_train, _y_train, random_state=11)

		sentences = _X_train[j*batch_size: (j+1) * batch_size]
		targets = _y_train[j*batch_size: (j+1) * batch_size]
		j += 1

		input_ids, input_masks = [],[]
		
		# see if puting following before the loop may improve the training in time and RAM used
		inputs = tokenizer.batch_encode_plus(list(sentences), add_special_tokens=True, max_length=max_length, 
											padding='max_length',  return_attention_mask=True,
											return_token_type_ids=True, truncation=True)

		ids = np.asarray(inputs['input_ids'], dtype='int32')
		masks = np.asarray(inputs['attention_mask'], dtype='int32')
		
		#till here and use the same shuffle on ids, masks instead of X_train
		
		inputs = [ids, masks]
		
		yield inputs, tf.one_hot(targets, depth=2) #targets

def load_transformer_models(bert, special_tokens):
	"""
	Objective: load the tokenizer we'll use and also the transfomer model
	
	Inputs:
		- bert, str: the name of models look at https://huggingface.co/models for all models
		- special_tokens, list: list of str, where they are tokens to be considered as one token
	Outputs:
		- tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
		- transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
																					we will use as base
																					(embedding model)
	"""
	tokenizer = AutoTokenizer.from_pretrained(bert)

	tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

	transformer_model = TFAutoModelForSequenceClassification.from_pretrained(bert)
	
	return tokenizer, transformer_model

def get_inputs(tokenizer, sentences, max_length):
    """
    Objective: tokenize the sentences to get the inputs
    
    Inputs:
        - tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
        - sentences, np.array: the sentences pre-processed to classify the intents
        - max_length, int: the maximum number of tokens
    Outputs:
        - inputs, list: list of ids and masks from the tokenizer
    """
    inputs = tokenizer.batch_encode_plus(list(sentences), add_special_tokens=True, max_length=max_length, 
                                    padding='max_length',  return_attention_mask=True,
                                    return_token_type_ids=True, truncation=True)

    ids = np.asarray(inputs['input_ids'], dtype='int32')
    masks = np.asarray(inputs['attention_mask'], dtype='int32')

    inputs = [ids, masks]
    
    return inputs

def train_bert(X_train, y_train, X_dev, Y_dev, tokenizer, batch_size, max_length, transformer_model, categories,lr, epochs):
  
  steps_per_epoch = int(len(X_train) / batch_size)
  batches = get_batches(X_train, y_train, tokenizer, batch_size, max_length)
  logging.info('Data batches generated')
  model = get_model(max_length, transformer_model, num_labels=len(categories), name_model=False)
  logging.info('Model loaded')
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0), metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
  model.fit(batches, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[callback], validation_data=(get_inputs(tokenizer, X_dev, max_length), tf.one_hot(Y_dev, depth=2)))
  
  return model
