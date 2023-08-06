import spacy

model_name = "en_core_web_sm"
if model_name not in spacy.util.get_installed_models():
    spacy.cli.download(model_name)
nlp = spacy.load(model_name, disable=['parser', 'ner'])


def filter_tokens(text: str, join: bool = False) -> str or list:
    """
    Filter out punctuation and stop tokens.
    :param text: input text (sentence)
    :param join: if True, return string, else return list of tokens
    :return: text without punctuation and stop words
    """
    doc = nlp(text)
    filtered = [token.lower_ for token in doc if not token.is_stop and not token.is_punct]
    if join:
        return ' '.join(filtered)
    return filtered


def extract_lemma(text: str, join: bool = False) -> str or list:
    """
    Extract lemma(base) from each word in the text.
    Example: feeling -> feel; pencils -> pencil; exhausted -> exhaust
    :param text: input text (sentence)
    :param join: if True, return string, else return list of tokens
    :return: text with each word replaced to its base word
    """
    doc = nlp(text)
    ret = [token.lemma_ for token in doc]
    if join:
        return ' '.join(ret)
    return ret


def filter_and_extract_lemma(text: str, join: bool = False) -> str or list:
    """
    Filter out punctuation and stop tokens and extract lemma(base) from each word in the text.
    :param text: input text (sentence)
    :param join: if True, return string, else return list of tokens
    :return: text without punctuation and stop words and each word replaced to its base word
    """
    doc = nlp(text)
    filtered = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    if join:
        return ' '.join(filtered)
    return filtered
