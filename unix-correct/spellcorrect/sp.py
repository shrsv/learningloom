import re
import ast
from collections import Counter

# ==========================================
# 1. DATA LOADING
# ==========================================

def load_corpus(filename='corpus.data'):
    """Load raw text and return a list of tokens."""
    return re.findall(r'[a-z]+', open(filename).read().lower())

def load_matrix(filename):
    """Load confusion matrix using ast.literal_eval."""
    return ast.literal_eval(open(filename).read())

# Load data files exactly as named in the directory
words = load_corpus('corpus.data')
add_m = load_matrix('addconfusion.data')
sub_m = load_matrix('subconfusion.data')
rev_m = load_matrix('revconfusion.data')
del_m = load_matrix('delconfusion.data')
raw_corpus = open('corpus.data').read().lower()

# ==========================================
# 2. N-GRAM MODELS
# ==========================================

def train_model(words):
    """Generate unigram and bigram counts."""
    return {
        'uni': Counter(words),
        'bi': Counter(zip(words, words[1:]))
    }

model = train_model(words)
VOCAB_SIZE = len(model['uni'])

# ==========================================
# 3. ERROR CLASSIFICATION (Original Logic)
# ==========================================

def get_error_details(correction, typo):
    """Identify the edit type and the character context (x,y) used for matrix lookup."""
    if correction == typo: return None, None

    # Deletion (in typo) means an insertion in the correction
    if len(correction) < len(typo):
        for i in range(len(correction)):
            if correction[i] != typo[i]:
                char_before = correction[i-1] if i > 0 else '#'
                return 'add', char_before + typo[i]
        return 'add', correction[-1] + typo[-1]

    # Insertion (in typo) means a deletion in the correction
    if len(correction) > len(typo):
        for i in range(len(typo)):
            if correction[i] != typo[i]:
                char_before = correction[i-1] if i > 0 else '#'
                return 'del', char_before + correction[i]
        return 'del', correction[-2] + correction[-1]

    # Substitution or Reversal
    for i in range(len(correction)):
        if correction[i] != typo[i]:
            if i < len(correction) - 1 and correction[i+1] == typo[i] and correction[i] == typo[i+1]:
                return 'rev', correction[i:i+2]
            return 'sub', correction[i] + typo[i]

    return None, None

# ==========================================
# 4. PROBABILITY CALCULATIONS
# ==========================================

def get_channel_prob(w, typo, raw_corpus):
    """P(typo|word) using confusion matrices and char/bigram counts from corpus."""
    etype, xy = get_error_details(w, typo)
    if not etype: return 1.0

    matrix = {'add': add_m, 'sub': sub_m, 'rev': rev_m, 'del': del_m}[etype]

    # The original logic uses the count of xy in the corpus for del/rev
    # and the count of x (the first char) for add/sub.
    if etype in ['del', 'rev']:
        denom = raw_corpus.count(xy)
    else:
        denom = raw_corpus.count(xy[0])

    return (matrix.get(xy, 0) + 1) / (denom + 10)

def get_prior_prob(word, prev, model):
    """P(word|prev) using Add-1 smoothing."""
    bigram_count = model['bi'].get((prev, word), 0)
    unigram_count = model['uni'].get(prev, 0)
    return (bigram_count + 1) / (unigram_count + VOCAB_SIZE)

# ==========================================
# 5. CORE SOLVER
# ==========================================

def edits1(word):
    """Generate all possible edits 1 distance away."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return set([L + R[1:] for L, R in splits if R] +
               [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1] +
               [L + c + R[1:] for L, R in splits if R for c in letters] +
               [L + c + R for L, R in splits for c in letters])

def correct_word(word, prev, model, raw_corpus):
    """Determine best correction by maximizing (P(word|prev) * P(typo|word))."""
    if word in model['uni']: return word

    candidates = [c for c in edits1(word) if c in model['uni']]
    if not candidates: return word

    # We multiply by 10^9 to avoid floating point underflow, matching original scale
    best_cand = max(candidates, key=lambda c:
                    get_prior_prob(c, prev, model) * get_channel_prob(c, word, raw_corpus) * 10**9)
    return best_cand

def correct_sentence(sentence, model, raw_corpus):
    """Correct sentence word by word."""
    tokens = re.findall(r'[a-z]+', sentence.lower())
    result = []
    for i, word in enumerate(tokens):
        prev = result[i-1] if i > 0 else ""
        result.append(correct_word(word, prev, model, raw_corpus))
    return " ".join(result)

# ==========================================
# 6. EXECUTION
# ==========================================

print("--- Running Spellcheck ---")
input_text = "she is a briliant acress"
output_text = correct_sentence(input_text, model, raw_corpus)

print(f"Input:    {input_text}")
print(f"Response: {output_text}")

