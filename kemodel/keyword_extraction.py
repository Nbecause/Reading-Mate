from keybert import KeyBERT
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os

from kiwipiepy import Kiwi
from flair.embeddings import TransformerDocumentEmbeddings
from sentence_transformers import SentenceTransformer

from utils import preprocessing_data, extracts_nouns, get_cosine_similarity
from keybert_model import KeywordExtraction

if __name__:
    
    # koquard 데이터 불러오기
    DATA_DIR = '../../data/squad_kor_v1_test_reformatted.csv'
    train_df = pd.read_csv(DATA_DIR)
    temp_df = train_df[['context', 'question', 'answer']]

    # context별로 id 재구성(같은 context -> 같은 id)
    temp_df['id'] = temp_df['context'].astype('category').cat.codes

    # unique한 context 데이터셋 만들기
    answer_dict = defaultdict()
    docs_list = []
    for _, data in tqdm(temp_df.iterrows(), desc='unique한 context 데이터셋 만들기', total = len(temp_df)):
        id = data['id']
        answer = data['answer']
        if id not in answer_dict.keys():
            answer_dict[id] = [answer]
            docs_list.append([id, preprocessing_data(data['context'])])
        else:
            answer_dict[id].append(answer)

    final_dict = defaultdict()
    for id, answer in answer_dict.items():
        final_dict[id] = list(set(answer))
    docs_df = pd.DataFrame(docs_list, columns=['id', 'context'])
    answer_df = pd.DataFrame(list(final_dict.items()), columns = ['id', 'answer'])
    
    # id, context, answer
    data_df = pd.merge(docs_df, answer_df, on='id')

    # KeyBERT options
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['sentence_transformer', 'flair'], help='모델 타입')
    parser.add_argument('--model_name', required=False, default='paraphrase-multilingual-MiniLM-L12-v2', help='모델 이름')
    parser.add_argument('--num_to_gen', required=False, type=int, default=5, help='생성할 keyword 수')
    parser.add_argument('--use_maxsum', required=True, default = True, help='다양성 방법1')
    parser.add_argument('--nr_candidates',required=False, default=3, type=int, help='use_maxsum=True일 경우 고려할 대상 개수')
    parser.add_argument('--use_mmr', required=True, default = True, help='다양성 방법2')
    parser.add_argument('--diversity', required=False, type=float, default=0.1, help='use_mmr=True할 경우 다양성을 얼마나 줄건지(숫자 클수록 다양성 커짐)')
    args = parser.parse_args()
    
    if args.model_type == 'sentence_transformer':
        print('--- sentence transformer ---')
        emb_model = SentenceTransformer(args.model_name)
        kw_model = KeyBERT(model=emb_model)
    elif args.model_type=='flair':
        print('--- flair ---')
        emb_model = TransformerDocumentEmbeddings(args.model_name)
        kw_model = KeyBERT(model=emb_model)

    keywords_object = KeywordExtraction(kw_model, 
                                        num_to_gen = args.num_to_gen, 
                                        stop_words = None,
                                        use_maxsum=args.use_maxsum, 
                                        nr_candidates=args.nr_candidates,
                                        use_mmr=args.use_mmr,
                                        diversity=args.diversity)
    
    # 키워드 추출
    new_data = []
    kiwi_model = Kiwi()
    for _, data in tqdm(data_df.iterrows(), desc='keyword extraction', total = len(data_df)):
        id = data['id']
        context = data['context']
        keywords_candidates = keywords_object.generate_keywords(context, 1)
        keywords_candidates = [extracts_nouns(kiwi_model, i) for i in keywords_candidates if (len(extracts_nouns(kiwi_model, i))>0) and (extracts_nouns(kiwi_model, i) in context)]
        new_data.append([id, context, list(set(keywords_candidates))])
    keyword_df = pd.DataFrame(new_data, columns=['id', 'context', 'keyword'])
    
    # id, context, answer, keyword
    answer_keyword_df = pd.merge(keyword_df, data_df, on=['id', 'context'])

    total_answer = sum(len(answer_keyword_df.iloc[i]['answer']) for i in range(len(answer_keyword_df)))
    total_keyword = sum(len(answer_keyword_df.iloc[i]['keyword']) for i in range(len(answer_keyword_df)))

    total_cs_score = 0
    score = 0

    cs_limit = 0.7
    for _, data in tqdm(answer_keyword_df.iterrows(), desc='점수 계산', total=len(answer_keyword_df)):
        id = data['id']
        answer_list = data['answer']
        for keyword in data['keyword']:
            cs_score = get_cosine_similarity(emb_model, args.model_type, args.model_name, keyword, answer_list)
            total_cs_score += cs_score
            if cs_limit <= cs_score:
                score += 1
    
    precision = score / total_keyword
    recall = score / total_answer

    f1_score = '{:.4f}'.format(2*precision*recall/(precision+recall))
    total_cs_score = '{:.4f}'.format(total_cs_score/total_keyword)
    score_dict = {
        'f1_score':f1_score,
        'average_cs_score':total_cs_score,
        'model_name':args.model_name, 
        'num_to_gen':args.num_to_gen,
        'use_maxsum':args.use_maxsum, 
        'nr_candidates':args.nr_candidates, 
        'use_mmr':args.use_mmr, 
        'diversity':args.diversity,
        }
    score_df = pd.DataFrame([score_dict])

    #keyword vs answer 비교 csv파일
    merged_df = pd.merge(keyword_df, answer_df, on='id', how='left')

    merged_df[['id', 'keyword', 'answer', 'context']].to_csv('keyword_answer.csv', index=False)

    file = 'score.csv'
    if os.path.isfile(file):
        with open(file, 'a') as file_data:
            score_df.to_csv(file, mode='a',index=False, header=False)
    else:
        score_df.to_csv(file, index=False)