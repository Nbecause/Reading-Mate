'''
    데이터셋의 형태를 통일된 형태로 바꿔주는 코드
'''
from datasets import load_dataset
import argparse
from tqdm import tqdm
from utils import find_sentence_at_index, change_column_name

# 옵션 arg 받기
parser = argparse.ArgumentParser()
parser.add_argument('--csvout', help='추출할 csv 파일 이름(확장자 빼고)')
parser.add_argument('--subset', default='train', help='train, test, 혹은 valid ')
parser.add_argument('--dataset', default='lmqg/qg_koquad', help='로컬/허깅페이스에서 가져올 데이터셋 이름')
args = parser.parse_args()

# 허깅페이스에서 데이터셋 가져오기
dataset = load_dataset(args.dataset)

# 판다스 데이터프레임으로 변환
train_data = dataset[args.subset].to_pandas()

# 데이터셋에 따라서 다른 처리를 해줘야 한다.
# KorQuad 1.0의 경우
if args.dataset == 'squad_kor_v1':
    col_name_dict = { 'id': 'id', 'answer' : 'answer', 'question':'question', 'context':'context',
                      'question_level':'question_level', 'question_type':'question_type',
                      'answer_type':'answer_type', 'answer_start':'answer_start', 'answer_end':'answer_end',
                      'clue_text':'clue_text', 'clue_start':'clue_start', 'clue_end':'clue_end' }
    # 칼럼 이름 바꾸거나 생성
    train_data = change_column_name(col_name_dict, train_data)
    
    # answer, answer_start, answer_end, clue_text, clue_start, clue_end 칼럼에 값 넣어주기(토큰단위가 아닌 글자단위 인덱스)
    for idx, row in tqdm(train_data.iterrows()):
        train_data.loc[idx,'answer'] = row.answers['text'][0]
        train_data.loc[idx,'answer_start'] = answer_start = row.answers['answer_start'][0]
        train_data.loc[idx,'answer_end'] = answer_start + len(row.answers['text'][0])
        clue_text, start_idx, end_idx = find_sentence_at_index(row.context, answer_start)
        train_data.loc[idx,'clue_text'] = clue_text
        train_data.loc[idx,'clue_start'] = start_idx
        train_data.loc[idx,'clue_end'] = end_idx

    # question_level, question_type, answer_type도 값을 넣어줄 수 있다면 넣어주자.
    pass
    # 불필요한 열 삭제 title, answers
    train_data.drop(['title', 'answers'], axis=1, inplace=True)

# QG_Koquad의 경우
if args.dataset == 'lmqg/qg_koquad':
    col_name_dict = { 'paragraph_id':'id', 'answer' : 'answer', 'question':'question', 'paragraph':'context',
                      'question_level':'question_level', 'question_type':'question_type',
                      'answer_type':'answer_type', 'answer_start':'answer_start', 'answer_end':'answer_end',
                      'sentence':'clue_text', 'clue_start':'clue_start', 'clue_end':'clue_end' }
    # 칼럼 이름 바꾸거나 생성
    train_data = change_column_name(col_name_dict, train_data)
    
    # answer_start, answer_end, clue_start, clue_end 칼럼에 값 넣어주기(토큰단위가 아닌 글자단위 인덱스)
    for idx, row in tqdm(train_data.iterrows()):
        clue_start = row.paragraph_sentence.find('<hl>')
        if clue_start != -1:
            train_data.loc[idx,'clue_start'] = clue_start
            train_data.loc[idx,'clue_end'] = clue_start + len(row.clue_text)
            train_data.loc[idx,'answer_start'] = answer_start = clue_start + row.clue_text.find(row.answer)
            train_data.loc[idx,'answer_end'] = answer_start + len(row.answer)

    # question_level, question_type, answer_type도 값을 넣어줄 수 있다면 넣어주자.
    pass
    # 불필요한 열 삭제 paragraph_question, sentence_answer, paragraph_answer, paragraph_sentence
    train_data.drop(['paragraph_question', 'sentence_answer', 'paragraph_answer', 'paragraph_sentence'], axis=1, inplace=True)

# csv 파일로 추출
if args.csvout is not None:
    train_data.to_csv(f'./{args.csvout}.csv')
else:
    print("출력할 csv 파일 이름을 넣어주지 않아서 아무것도 출력되지 않습니다")