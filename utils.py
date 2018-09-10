from nltk.corpus import stopwords


EPS = 1e-45

STOP_WORDS = set(stopwords.words('english'))

PUNCT = {
    '.', ',', '-', '--', ';', ':',  # want to keep `!` and `?` because these are semantic
    "``", "'", '`', '""', "''",
    '...',  '-lrb-', '-rrb-'
}
