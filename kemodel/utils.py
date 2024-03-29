import re
import numpy as np

from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from kiwipiepy import Kiwi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def make_embeddings(emb_model, model_type, model_name, keyword, answers_list):
    """ Convert keyword and answers to embedding vector

    Args:
        emb_model : embedding model
        model_type (string): embedding model type
        model_name (string): embedding model name
        keyword (string): keyword
        answers_list (list): answers list

    Returns:
        keyword와 answers의 embedding vector를 반환
    """
    if model_type == 'sentence_transformer':
        keyword_emb = emb_model.encode(keyword)
        answer_emb = emb_model.encode(answers_list)

    elif model_type == 'flair':
        emb_model = TransformerDocumentEmbeddings(model_name)
        keyword = Sentence(keyword)
        emb_model.embed(keyword)
        keyword_emb = keyword.embedding.detach().cpu().numpy()

        answer_emb = []
        for answer in answers_list:
            answer = Sentence(answer)
            emb_model.embed(answer)
            answer_emb.append(answer.embedding.detach().cpu().numpy())
    return [keyword_emb], answer_emb

def preprocessing_data(context):
    '''
    html tag 제거, 공백 하나로 대체
    '''
    temp_context = re.sub(r'<[^>]+>', ' ', context)
    temp_context = re.sub(r'\s+', ' ', temp_context)
    final_context = temp_context.lower()
    return final_context

def extracts_nouns(kiwi_model, keyword):
    """ keyword에서 명사만 추출해 반환

    Args:
        kiwi_model: 한국어 라이브러리
        keyword (string): keyword

    Returns:
        string: 명사형 keyword
    """
    kiwi = kiwi_model
    final_keyword = []
    for k in keyword.split():
        temp_keyword = ''
        for i in kiwi.tokenize(k):
            if i.tag.startswith('N') or i.tag=='SN' and i.tag !='NP':
                temp_keyword += i.form
        final_keyword.append(temp_keyword)
    if len(final_keyword) != 0:
        return ' '.join(final_keyword)

def get_cosine_similarity(emb_model, model_type, model_name, keyword, answers_list):
    """ keyword와 각 answer 사이의 cosine similarity 중 가장 큰 값 반환

    Args:
        emb_model : embedding model
        model_type (string): embedding model type
        model_name (string): embedding model name
        keyword (string): keyword
        answers_list (list): answers list

    Returns:
        int: 가장 높은 cosine similarity 값
    """
    keyword_emb, answers_embs = make_embeddings(emb_model, model_type, model_name, keyword, answers_list)
    cosine_list = cosine_similarity(keyword_emb, answers_embs)
    return cosine_list.max()