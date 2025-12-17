"""Functional-style Spell Correction (Norvig style)

A functional approach to correcting non-word spelling errors using n-gram MAP Language Models,
Noisy Channel Model, Error Confusion Matrix, and Damerau-Levenshtein Edit Distance.

Includes integrated n-gram language model implementation.

Usage:
>>> context = load_spell_context()
>>> result = correct_sentence('she is a briliant acress', context)
>>> print(result)
'she is a brilliant actress'
"""

import ast
import math
from collections import Counter

# ============================================================================
# N-GRAM LANGUAGE MODEL FUNCTIONS
# ============================================================================

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

# ============================================================================
# SPELL CORRECTION FUNCTIONS
# ============================================================================

# Data loading functions
def load_dictionary(filename='dictionary.data'):
    """Load dictionary from external data file."""
    print(f"Loading dictionary from {filename}")
    with open(filename, 'r') as f:
        return f.read().split("\n")

def load_confusion_matrices():
    """Load all confusion matrices from data files."""
    print("Loading confusion matrices")
    matrices = {}
    
    for matrix_type in ['add', 'sub', 'rev', 'del']:
        filename = f"{matrix_type}confusion.data"
        with open(filename, 'r') as f:
            matrices[matrix_type] = ast.literal_eval(f.read())
    
    return matrices

def load_spell_context(corpus_file='corpus.data'):
    """Load all context needed for spell correction: models, dictionary, confusion matrices."""
    print("Loading spell correction context...")
    models = create_ngrams(['uni', 'bi'], corpus_file=corpus_file)
    words = sorted(set(models['words']))[3246:]  # Skip first 3246 for efficiency
    dictionary = load_dictionary()
    matrices = load_confusion_matrices()
    
    return {
        'models': models,
        'words': words,
        'dictionary': dictionary,
        'matrices': matrices,
        'corpus': ' '.join(models['words'])
    }

# Edit distance calculation
def damerau_levenshtein_distance(s1, s2):
    """Calculate Damerau-Levenshtein Edit Distance between two strings."""
    s1, s2 = '#' + s1, '#' + s2
    m, n = len(s1), len(s2)
    
    # Initialize distance matrix
    D = [[0] * n for _ in range(m)]
    for i in range(m):
        D[i][0] = i
    for j in range(n):
        D[0][j] = j
    
    # Calculate distances
    for i in range(1, m):
        for j in range(1, n):
            costs = [
                D[i-1][j] + 1,                    # deletion
                D[i][j-1] + 1,                    # insertion
                D[i-1][j-1] + (2 if s1[i] != s2[j] else 0)  # substitution
            ]
            
            # Transposition
            if i > 1 and j > 1 and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                costs.append(D[i-2][j-2] + 1)
            
            D[i][j] = min(costs)
    
    return D[m-1][n-1]

# Candidate generation
def generate_candidates(word, words, max_distance=1):
    """Generate candidate corrections within edit distance threshold."""
    candidates = {}
    for w in words:
        distance = damerau_levenshtein_distance(word, w)
        if distance <= max_distance:
            candidates[w] = distance
    return sorted(candidates, key=candidates.get)

# Edit type detection
def detect_edit_type(candidate, word):
    """Detect the type of edit (insertion, deletion, substitution, reversal) between strings."""
    
    def check_edits(cand, wrd, reverse=False):
        """Helper to check edits in forward or reverse direction."""
        for i in range(min(len(wrd), len(cand)) - 1):
            if cand[0:i+1] != wrd[0:i+1]:
                # Deletion
                if cand[i:] == wrd[i-1:]:
                    correct = cand[i-1]
                    x = cand[i-2] if i > 1 else ''
                    w = x + correct
                    return ("Deletion", correct, '', x, w)
                
                # Insertion
                elif cand[i:] == wrd[i+1:]:
                    error = wrd[i]
                    if i == 0:
                        w, x = '#', '#' + error
                    else:
                        w, x = wrd[i-1], wrd[i-1] + error
                    return ("Insertion", '', error, x, w)
                
                # Substitution
                if i + 1 < len(cand) and i + 1 < len(wrd) and cand[i+1:] == wrd[i+1:]:
                    correct, error = cand[i], wrd[i]
                    return ("Substitution", correct, error, error, correct)
                
                # Reversal (transposition)
                if (i + 1 < len(wrd) and i + 2 <= len(cand) and 
                    cand[i] == wrd[i+1] and i + 2 <= len(wrd) and cand[i+2:] == wrd[i+2:]):
                    correct = cand[i] + cand[i+1]
                    error = wrd[i] + wrd[i+1]
                    return ("Reversal", correct, error, error, correct)
        
        return None
    
    if word == candidate:
        return ("None", '', '', '', '')
    
    # Try forward
    result = check_edits(candidate, word)
    if result:
        return result
    
    # Try reverse
    result = check_edits(candidate[::-1], word[::-1], reverse=True)
    if result:
        return result
    
    return ("None", '', '', '', '')

# Channel model probability
def channel_model_probability(x, y, edit_type, matrices, corpus):
    """Calculate channel model probability for errors using confusion matrices."""
    if edit_type == 'add':
        if x == '#':
            count = corpus.count(' ' + y)
        else:
            count = corpus.count(x)
        return matrices['add'].get(x + y, 0) / max(count, 1)
    
    elif edit_type == 'sub':
        key = (x + y)[0:2]
        count = corpus.count(y)
        return matrices['sub'].get(key, 0) / max(count, 1)
    
    elif edit_type == 'rev':
        count = corpus.count(x + y)
        return matrices['rev'].get(x + y, 0) / max(count, 1)
    
    elif edit_type == 'del':
        count = corpus.count(x + y)
        return matrices['del'].get(x + y, 0) / max(count, 1)
    
    return 0.0

# Word correction
def correct_word(word, prev_word, next_word, context):
    """Correct a single word using noisy channel model."""
    candidates = generate_candidates(word, context['words'])
    
    # If word is already valid, return it
    if word in candidates:
        return word
    
    # Calculate probabilities for each candidate
    scores = {}
    
    for candidate in candidates:
        edit_info = detect_edit_type(candidate, word)
        
        if edit_info[0] == "None":
            continue
        
        edit_type_map = {
            "Insertion": ('add', edit_info[3][0] if len(edit_info[3]) > 0 else '', 
                         edit_info[3][1] if len(edit_info[3]) > 1 else ''),
            "Deletion": ('del', edit_info[4][0] if len(edit_info[4]) > 0 else '', 
                        edit_info[4][1] if len(edit_info[4]) > 1 else ''),
            "Reversal": ('rev', edit_info[4][0] if len(edit_info[4]) > 0 else '', 
                        edit_info[4][1] if len(edit_info[4]) > 1 else ''),
            "Substitution": ('sub', edit_info[3], edit_info[4])
        }
        
        if edit_info[0] not in edit_type_map:
            continue
        
        edit_type, x, y = edit_type_map[edit_info[0]]
        channel_prob = channel_model_probability(x, y, edit_type, 
                                                 context['matrices'], context['corpus'])
        
        # Calculate language model probability using bigrams
        if next_word:
            phrase = f"{prev_word} {candidate} {next_word}" if prev_word else f"{candidate} {next_word}"
        else:
            phrase = f"{prev_word} {candidate}" if prev_word else candidate
        
        try:
            lm_prob = math.exp(sentence_probability(phrase, context['models'], 'bi', 'log'))
        except:
            lm_prob = 1e-10
        
        # Combine probabilities (noisy channel model)
        scores[candidate] = channel_prob * lm_prob * 1e9
    
    # Return best candidate or empty string if none found
    if scores:
        return max(scores, key=scores.get)
    return ''

# Sentence correction
def correct_sentence(sentence, context):
    """Correct spelling errors in a sentence."""
    words = sentence.lower().split()
    corrected = []
    
    for i, word in enumerate(words):
        prev_word = words[i-1] if i > 0 else ''
        next_word = words[i+1] if i < len(words) - 1 else ''
        
        corrected_word = correct_word(word, prev_word, next_word, context)
        corrected.append(corrected_word if corrected_word else word)
    
    return ' '.join(corrected)

# Interactive spell correction
def interactive_spell_check(context=None):
    """Run interactive spell checking loop."""
    if context is None:
        context = load_spell_context()
    
    print("\nInteractive Spell Checker (Norvig Style)")
    print("Enter sentences to correct (Ctrl+C to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input('\nInput: ').strip()
            if not user_input:
                continue
            
            corrected = correct_sentence(user_input, context)
            print(f'Response: {corrected}')
            
        except KeyboardInterrupt:
            print("\n\nExiting spell checker. Goodbye!")
            break
        except EOFError:
            break

# Main execution
if __name__ == '__main__':
    # Example usage
    # context = load_spell_context()
    # result = correct_sentence('she is a briliant acress', context)
    # print(result)
    
    # Run interactive mode
    interactive_spell_check()
