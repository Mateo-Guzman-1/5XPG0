"""The keyword the whole pipeline detects, and its per-keyword data.

Set it with the environment variable KWS_KEYWORD (default "yes", the release
keyword). It must be one of the 35 Speech Commands words, so that stage 1
(35 words, keyword-agnostic) and the live test sets exist for it. Every
stream checkpoint and integer model records its keyword; the detectors refuse
a model trained for another one.

Per keyword:
  NEAR_MISS     real words fetched from MSWC as hard negatives and grouped in
                the evaluation (robust_eval.MSWC_GROUPS)
  TTS_GROUPS    the synthetic-word probe groups (robust_eval tts)
  MULTI         the data/multi* folder built by prepare_multicorpus.py
  prefixed(w)   words that begin with the complete keyword (yesterday,
                sheila's): "don't care" for a causal detector (JOURNAL entry 21)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYWORD = os.environ.get('KWS_KEYWORD', 'yes').strip().lower()

NEAR_MISS = {
    'yes': ('yet', 'yeah', 'yesterday', 'yellow', 'yell', 'yep', 'year', 'years', 'young', 'you',
            'guess', 'less', 'mess', 'bless', 'dress', 'press', 'chess', 'address', 'unless', 'success',
            'says', 'this', 'us', 'plus', 'yes', 'its', 'gets', 'lets', 'sets', 'bets', 'jets', 'pets',
            'eats', 'meets', 'seats', 'streets', 'each', 'reach', 'beach', 'peach', 'teach', 'speech',
            'fetch', 'sketch', 'stretch', 'check', 'jess', 'tess', 'nest', 'best', 'rest', 'west', 'test',
            'said', 'set', 'sex', 'next', 'ex', 'text', 'pizza', 'jazz', 'cheese', 'ease', 'these'),
    # /ʃ iː l ə/: the /ʃiː/ onset, the /iːlə/ rhyme, and -la/-ler endings.
    'sheila': ('she', 'sheet', 'sheep', 'sheen', 'sheer', 'shield', 'shields', 'sheila',
               'she\'ll', 'shell', 'shelf', 'shelly', 'shelley', 'shiny', 'cheap', 'cheese', 'chief', 'chiefly',
               'leila', 'lila', 'layla', 'delilah', 'tequila', 'godzilla', 'gorilla', 'vanilla', 'priscilla',
               'stella', 'bella', 'ella', 'della', 'celia', 'sheena', 'shana', 'shayla', 'sealer', 'feeler',
               'healer', 'dealer', 'ceiling', 'feeling', 'healing', 'sealing', 'she\'s', 'shelter', 'shelby'),
}
TTS_GROUPS = {
    'yes': {'yes': ['yes'], 'yeets/yets/yetz': ['yeets', 'yets', 'yetz'], 'pizza(s)': ['pizza', 'pizzas'],
            'eats/its': ['eats', 'its'], 'other -ts': ['jets', 'gets', 'bets', 'lets', 'sets'],
            'ch words': ['yech', 'yetch', 'each', 'peach'], 'yeah': ['yeah'], 'yeet': ['yeet'],
            'guess/less': ['guess', 'less'], 'cheese': ['cheese']},
    'sheila': {'sheila': ['sheila'], 'she/sheet/sheep': ['she', 'sheet', 'sheep'],
               'shield/shelf/shell': ['shield', 'shelf', 'shell'], 'leila/lila/layla': ['leila', 'lila', 'layla'],
               'tequila/vanilla/gorilla': ['tequila', 'vanilla', 'gorilla'], 'stella/bella/ella': ['stella', 'bella', 'ella'],
               'dealer/healer/feeler': ['dealer', 'healer', 'feeler'], 'sheena/shana/shayla': ['sheena', 'shana', 'shayla']},
}
if KEYWORD not in NEAR_MISS:
    raise ValueError(f'KWS_KEYWORD={KEYWORD!r}: add its NEAR_MISS and TTS_GROUPS to keyword_config.py')

MULTI = ROOT / ('data/multi' if KEYWORD == 'yes' else f'data/multi_{KEYWORD}')
if os.environ.get('KWS_MULTI'):   # e.g. data/multi_sheila_full while another build is in use
    MULTI = Path(os.environ['KWS_MULTI']) if Path(os.environ['KWS_MULTI']).is_absolute() else ROOT / os.environ['KWS_MULTI']
TTS = ROOT / ('data/tts' if KEYWORD == 'yes' else f'data/tts_{KEYWORD}')   # make_tts_negatives.py output

# MSWC near-miss groups for the evaluation (the prefixed group is added by robust_eval).
MSWC_GROUPS = {
    'yes': {
        'ye- (yet, yeah, yep, yell, yellow)': ['yet', 'yeah', 'yep', 'yell', 'yellow'],
        '/s/-final (guess, less, this, us, ...)': ['guess', 'less', 'mess', 'bless', 'dress', 'press', 'chess',
                                                  'address', 'unless', 'success', 'says', 'this', 'us', 'plus',
                                                  'jess', 'tess'],
        '/ts/-final (its, gets, lets, eats, ...)': ['its', 'gets', 'lets', 'sets', 'bets', 'jets', 'pets', 'eats',
                                                   'meets', 'seats', 'streets'],
        '/tʃ/-final (each, reach, speech, ...)': ['each', 'reach', 'beach', 'peach', 'teach', 'speech', 'fetch',
                                                 'sketch', 'stretch'],
        '/st/, /ks/ (best, test, next, ...)': ['nest', 'best', 'rest', 'west', 'test', 'sex', 'next', 'ex', 'text'],
        'other near (year, you, said, cheese, ...)': ['year', 'years', 'young', 'you', 'said', 'set', 'check',
                                                      'pizza', 'jazz', 'cheese', 'ease', 'these'],
    },
    'sheila': {
        '/ʃiː/ onset (she, sheet, shield, ...)': ['she', 'sheet', 'sheep', 'sheen', 'sheer', 'shield', 'shields',
                                                 "she'll", "she's", 'shiny'],
        '/ʃɛ/ (shell, shelf, shelter, ...)': ['shell', 'shelf', 'shelly', 'shelley', 'shelter', 'shelby'],
        '/tʃiː/ (cheap, cheese, chief)': ['cheap', 'cheese', 'chief', 'chiefly'],
        '-ila/-ela names (leila, delilah, stella, ...)': ['leila', 'lila', 'layla', 'delilah', 'stella', 'bella',
                                                         'ella', 'della', 'celia', 'sheena', 'shana', 'shayla'],
        '-illa (tequila, vanilla, gorilla, ...)': ['tequila', 'godzilla', 'gorilla', 'vanilla', 'priscilla'],
        '-iːlər/-iːlɪŋ (dealer, feeling, ceiling, ...)': ['sealer', 'feeler', 'healer', 'dealer', 'ceiling',
                                                         'feeling', 'healing', 'sealing'],
    },
}[KEYWORD]


def mswc_table(base, split):
    """MSWC selection table of a split: <split>_selected.csv for "yes", <split>_selected_<kw>.csv otherwise."""
    return Path(base) / (f'{split}_selected.csv' if KEYWORD == 'yes' else f'{split}_selected_{KEYWORD}.csv')


def sc_keyword_labels(names):
    """1 for Speech Commands clips of the keyword (names are '<word>/<file>.wav'), else 0."""
    import numpy as np
    return np.array([str(n).split('/')[0] == KEYWORD for n in names], np.int64)


# Other spellings of the keyword in corpus transcripts: never negatives.
ALIASES = {'yes': (), 'sheila': ('shiela', 'sheilah', 'sheela')}[KEYWORD]


def excluded_negative(word):
    """Words that must never be sampled as negatives: prefixed forms and spellings of the keyword."""
    return prefixed(word) or str(word).lower() in ALIASES


def prefixed(word):
    """A word that begins with the complete keyword but is longer (yesterday, sheila's)."""
    w = str(word).lower()
    return w.startswith(KEYWORD) and w != KEYWORD


def check_model_keyword(recorded, what='model'):
    """Refuse a model trained for another keyword (models from before this module are "yes")."""
    rec = str(recorded if recorded is not None else 'yes').lower()
    if rec != KEYWORD:
        raise ValueError(f'{what} was trained for {rec!r}, but KWS_KEYWORD={KEYWORD!r}')
