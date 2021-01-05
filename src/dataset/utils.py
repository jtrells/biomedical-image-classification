from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.tokenize import word_tokenize
# from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
# nltk.download('punkt')

stopwords_pubmed = ['a', 'about', 'again', 'all', 'almost', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'but', 'by', 'can', 'could', 'did', 'do', 'does', 'done', 'due', 'during', 'each', 'either', 'enough', 'especially', 'etc', 'for', 'found', 'from', 'further', 'had', 'has', 'have', 'having', 'here', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'kg', 'km', 'made', 'mainly', 'make', 'may', 'mg', 'might', 'ml', 'mm', 'most', 'mostly', 'must', 'nearly', 'neither', 'no', 'nor', 'obtained', 'of', 'often', 'on', 'our', 'overall', 'perhaps', 'pmid', 'quite', 'rather', 'really', 'regarding', 'seem', 'seen', 'several', 'should', 'show', 'showed', 'shown', 'shows', 'significantly', 'since', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'therefore', 'these', 'they', 'this', 'those', 'through', 'thus', 'to', 'upon', 'use', 'used', 'using', 'various', 'very', 'was', 'we', 'were', 'what', 'when', 'which', 'while', 'with', 'within', 'without', 'would']

def clean_str(sentence):
    tokens = word_tokenize(sentence.lower())
    stopwords_english = stopwords.words('english')
    
    cleaned_sentence = []
    for word in tokens:
        if (word not in stopwords_english and word not in string.punctuation and word not in stopwords_pubmed):
            cleaned_sentence.append(word)
    cleaned_str = ' '.join(cleaned_sentence)
    cleaned_str = cleaned_str[:-2] if cleaned_str[-1] == '.' else cleaned_str
    return cleaned_str
