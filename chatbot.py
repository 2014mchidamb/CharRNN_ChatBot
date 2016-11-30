from keras.models import load_model
import numpy as np
import pickle

def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


char_indices = pickle.load(open('char_indices.p', 'rb'))
indices_char = pickle.load(open('indices_char.p', 'rb'))

model = load_model('char_rnn.h5')

cur = ''
prev = ''
response_length = 100
while cur != 'q':
	cur = raw_input('You: ').lower()
	prev = (prev+cur)[-40:]
	if len(prev) < 40:
		prev = ' '*(40-len(prev))+prev
	response = ''
	for i in range(response_length):
		x = np.zeros((1, len(prev), len(char_indices)))
		for t, char in enumerate(prev):
			x[0, t, char_indices[char]] = 1.
		preds = model.predict(x, verbose=0)[0]
		next_char = indices_char[sample(preds, temperature=0.5)]
		response += next_char
		prev = prev[1:]+next_char
	print "Bot: "+response
