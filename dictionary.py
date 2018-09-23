def get_stop_words():
    stop_words = [line.rstrip('\n') for line in open('dictionary/stop_words.txt')]
    return stop_words

def get_slang_words():
    slang_words = [line.rstrip('\n') for line in open('dictionary/slang_words.txt')]
    slang_words_keys = [key.split(':')[0] for key in slang_words]
    slang_words_values = [key.split(':')[1] for key in slang_words]
    slang_words = dict(zip(slang_words_keys, slang_words_values))
    return slang_words

def get_base_words():
    base_words = [line.rstrip('\n') for line in open('dictionary/base_words.txt')]
    return base_words