import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem import SnowballStemmer

# Utilizamos SnowballStemmer para el idioma español
stemmer = SnowballStemmer('spanish')

def tokenize(sentence):
    """
    Divide la oración en una lista de palabras/tokens.
    Un token puede ser una palabra, un carácter de puntuación o un número.
    """
    return nltk.word_tokenize(sentence, language='spanish')

def stem(word):
    """
    Aplica el stemming para encontrar la forma raíz de la palabra.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Retorna un arreglo de "bolsa de palabras":
    1 para cada palabra conocida que existe en la oración, 0 en caso contrario.
    """
    # Aplica stemming a cada palabra
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Inicializa la bolsa con 0 para cada palabra
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
