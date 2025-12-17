"""Functional-style n-Gram Language Model (Norvig style)

A functional approach to creating n-Gram (1-5) Maximum Likelihood 
Probabilistic Language Model with Laplace Add-1 smoothing.

Usage:
>>> models = create_ngrams(['uni', 'bi', 'tri'])
>>> prob = sentence_probability('hold your horses', models, 'bi', 'log')
>>> print(prob)
"""

from collections import Counter
import math

# Corpus loading and processing
def load_corpus(filename='corpus.data'):
    """Load and tokenize corpus from file."""
    print(f"Loading Corpus from {filename}")
    with open(filename, 'r') as f:
        corpus = f.read()
    print("Processing Corpus")
    return corpus.split(' ')

# N-gram creation functions
def create_unigram(words):
    """Create Unigram Model from word list."""
    print("Creating Unigram Model")
    return Counter(words)

def create_bigram(words):
    """Create Bigram Model from word list."""
    print("Creating Bigram Model")
    return Counter(' '.join(words[i:i+2]) for i in range(len(words)-1))

def create_trigram(words):
    """Create Trigram Model from word list."""
    print("Creating Trigram Model")
    return Counter(' '.join(words[i:i+3]) for i in range(len(words)-2))

def create_quadrigram(words):
    """Create Quadrigram Model from word list."""
    print("Creating Quadrigram Model")
    return Counter(' '.join(words[i:i+4]) for i in range(len(words)-3))

def create_pentigram(words):
    """Create Pentigram Model from word list."""
    print("Creating Pentigram Model")
    return Counter(' '.join(words[i:i+5]) for i in range(len(words)-4))

# Main ngram creation function
def create_ngrams(gram_types=['uni', 'bi'], words=None, corpus_file='corpus.data'):
    """Create specified n-gram models.
    
    Args:
        gram_types: list of gram types to create ['uni', 'bi', 'tri', 'quadri', 'penti']
        words: pre-loaded word list (if None, loads from corpus_file)
        corpus_file: path to corpus file
    
    Returns:
        Dictionary containing requested n-gram models and words
    """
    if words is None:
        words = load_corpus(corpus_file)
    
    models = {'words': words}
    
    creators = {
        'uni': create_unigram,
        'bi': create_bigram,
        'tri': create_trigram,
        'quadri': create_quadrigram,
        'penti': create_pentigram
    }
    
    for gram_type in gram_types:
        if gram_type in creators:
            models[gram_type] = creators[gram_type](words)
    
    return models

# Probability calculation
def probability(word, context="", gram_type='uni', models=None):
    """Calculate Maximum Likelihood Probability with Laplace smoothing.
    
    Args:
        word: single word for unigram, or first word(s) for higher n-grams
        context: the full n-gram string for bi/tri/quadri/penti-grams
        gram_type: 'uni', 'bi', 'tri', 'quadri', or 'penti'
        models: dictionary containing ngram models and words
    
    Returns:
        Log probability
    """
    if models is None:
        raise ValueError("models dictionary required")
    
    words = models['words']
    vocab_size = len(models.get('uni', Counter()))
    
    if gram_type == 'uni':
        return math.log((models['uni'][word] + 1) / (len(words) + vocab_size))
    
    elif gram_type == 'bi':
        bigram_count = models['bi'][context] + 1
        unigram_count = models['uni'][word] + vocab_size
        return math.log(bigram_count / unigram_count)
    
    elif gram_type == 'tri':
        trigram_count = models['tri'][context] + 1
        bigram_count = models['bi'][word] + vocab_size
        return math.log(trigram_count / bigram_count)
    
    elif gram_type == 'quadri':
        quadrigram_count = models['quadri'][context] + 1
        trigram_count = models['tri'][word] + vocab_size
        return math.log(quadrigram_count / trigram_count)
    
    elif gram_type == 'penti':
        pentigram_count = models['penti'][context] + 1
        quadrigram_count = models['quadri'][word] + vocab_size
        return math.log(pentigram_count / quadrigram_count)

# Sentence probability calculation
def sentence_probability(sentence, models, gram_type='uni', form='antilog'):
    """Calculate cumulative n-gram probability for a sentence.
    
    Args:
        sentence: input sentence/phrase
        models: dictionary containing ngram models
        gram_type: 'uni', 'bi', 'tri', 'quadri', or 'penti'
        form: 'log' for log probability, 'antilog' for regular probability
    
    Returns:
        Sentence probability (log or antilog form)
    """
    words = sentence.lower().split()
    n = len(words)
    
    # Define n-gram size
    gram_sizes = {'uni': 1, 'bi': 2, 'tri': 3, 'quadri': 4, 'penti': 5}
    gram_size = gram_sizes.get(gram_type, 1)
    
    if n < gram_size:
        return 0 if form == 'log' else 1
    
    # Calculate log probability sum
    log_prob = 0
    
    if gram_type == 'uni':
        log_prob = sum(probability(w, "", 'uni', models) for w in words)
    
    elif gram_type == 'bi':
        for i in range(n - 1):
            context = f"{words[i]} {words[i+1]}"
            log_prob += probability(words[i], context, 'bi', models)
    
    elif gram_type == 'tri':
        for i in range(n - 2):
            prefix = f"{words[i]} {words[i+1]}"
            context = f"{words[i]} {words[i+1]} {words[i+2]}"
            log_prob += probability(prefix, context, 'tri', models)
    
    elif gram_type == 'quadri':
        for i in range(n - 3):
            prefix = f"{words[i]} {words[i+1]} {words[i+2]}"
            context = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
            log_prob += probability(prefix, context, 'quadri', models)
    
    elif gram_type == 'penti':
        for i in range(n - 4):
            prefix = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
            context = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]} {words[i+4]}"
            log_prob += probability(prefix, context, 'penti', models)
    
    return log_prob if form == 'log' else math.exp(log_prob)

# Helper function for quick model creation and usage
def quick_model(gram_types=['uni', 'bi'], corpus_file='corpus.data'):
    """Quick helper to create models in one call."""
    return create_ngrams(gram_types, corpus_file=corpus_file)

# Example usage
if __name__ == '__main__':
    # Example: Create models and calculate probability
    # models = create_ngrams(['uni', 'bi', 'tri'])
    # prob = sentence_probability('hold your horses', models, 'bi', 'log')
    # print(f"Probability: {prob}")
    pass
