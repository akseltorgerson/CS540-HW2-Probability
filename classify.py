import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    doc = open(filepath, 'r')
    for word in doc:
        word = word.strip()
        if word in vocab:
            if not word in bow:
                bow[word] = 1
            else:
                bow[word] += 1
        else:
            if not None in bow:
                bow[None] = 1
            else:
                bow[None] += 1
    return bow

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1 # smoothing factor
    logprob = {}
    labelDict = {}

    for label in label_list:
        if not label in labelDict:
            labelDict[label] = smooth
    
    for data in training_data:
        currLabel = data['label']
        labelDict[currLabel] += 1

    for label in label_list:
        logprob[label] = math.log(labelDict[label] / (len(training_data) + 2))

    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    wordCnt = 0
    word_prob = {}

    for word in vocab:
        word_prob[word] = smooth
    
    word_prob[None] = smooth

    for data in training_data:
        if data['label'] == label:
            for word in data['bow']:
                wordCnt += data['bow'][word]
                if word in vocab:
                    word_prob[word] += data['bow'][word]
                else:
                    word_prob[None] += data['bow'][word]

    for word in word_prob:
        word_prob[word] = math.log(word_prob[word]) - math.log(wordCnt + smooth * (len(vocab) + 1))

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    prior_data = prior(training_data, label_list)

    retval['vocabulary'] = vocab
    retval['log prior'] = prior_data
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    doc = open(filepath, 'r', encoding='utf-8')
    for line in doc:
        line = line.strip()
        if line in model['vocabulary']:
            model['log prior']['2020'] += model['log p(w|y=2020)'][line]
            model['log prior']['2016'] += model['log p(w|y=2016)'][line]
        else:
            model['log prior']['2020'] += model['log p(w|y=2020)'][None]
            model['log prior']['2016'] += model['log p(w|y=2016)'][None]

    retval['log p(y=2020|x)'] = model['log prior']['2020']
    retval['log p(y=2016|x)'] = model['log prior']['2016']

    if model['log prior']['2020'] > model['log prior']['2016']:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    
    return retval
