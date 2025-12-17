"""Functional-style Spell Correction (Norvig style)

A functional approach to correcting non-word spelling errors using n-gram MAP Language Models,
Noisy Channel Model, Error Confusion Matrix, and Damerau-Levenshtein Edit Distance.

Usage:
>>> context = load_spell_context()
>>> result = correct_sentence('she is a briliant acress', context)
>>> print(result)
'she is a brilliant actress'
"""

import ast
import math
from ngram_fn import create_ngrams, sentence_probability

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
