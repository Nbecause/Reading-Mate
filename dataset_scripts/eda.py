from datasets import load_dataset
from transformers import AutoTokenizer
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 옵션 arg 받기
parser = argparse.ArgumentParser()
parser.add_argument('--csvout', help='추출할 csv 파일 이름')
parser.add_argument('--subset', default='train', help='train, test, 혹은 valid ')
parser.add_argument('--column', default='paragraph', help='분석할 칼럼의 이름')
parser.add_argument('--dataset', default='lmqg/qg_koquad', help='로컬/허깅페이스에서 가져올 데이터셋 이름')
parser.add_argument('--localfiletype', help='만약 데이터셋이 로컬파일이라면 이 인수에 확장자를 다시 적어주세요')
args = parser.parse_args()



# 로컬에서 데이터셋 가져오기 
if(args.localfiletype is not None):
    dataset = load_dataset(args.localfiletype, data_files=args.dataset)
    print(f'***************{args.dataset} 분석***************\n')
# 허깅페이스에서 데이터셋 가져오기
else:
    dataset = load_dataset(args.dataset)
    print(f'***************{args.dataset}의 {args.subset} 분석***************\n')

# 현재로서는 로컬은 csv 파일 타입만 상정하고 있음. 기본적으로 load_dataset을 쓰면 DataDict 타입 객체의 train에 넣어지더라.
assert(args.localfiletype is None or args.localfiletype =='csv')

# 데이터셋의 크기
if(args.localfiletype == 'csv'):
    print(f'{args.dataset}에 들어있는 데이터의 갯수는', len(dataset[args.subset]))
else:
    print(f'{args.subset}에 들어있는 데이터의 갯수는', len(dataset[args.subset]))
    # print('validation에 들어있는 데이터의 갯수는', len(dataset['validation']))
    # print('test에 들어있는 데이터의 갯수는', len(dataset['test']))

#판다스 데이터프레임으로 변환
if(args.localfiletype == 'csv'):
    train_data = dataset[args.subset].to_pandas()
else:
    train_data = dataset[args.subset].to_pandas()

# csv 파일로 추출
if(args.csvout is not None):
    train_data.to_csv(f'./{args.csvout}.csv')


# 띄어쓰기 단위 지문 길이 출력 
def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    
    return percentile_

print()
print("단어 단위 지문 길이(95 percentile) =",\
      train_data[args.column].apply(lambda x: len(x.split(' '))).agg([percentile(0.95)]).values[0])

train_data['len'] = train_data[args.column].apply(lambda x: len(x.split(' ')))
print(train_data.agg({'len':['min', 'max', 'mean']}))

# 그래프 출력
print()
print('지문 길이의 분포')
bins = list(np.arange(0, train_data['len'].max(), train_data['len'].max()/10).astype(int))
sentence_len_categories = pd.cut(train_data["len"], bins)
print(sentence_len_categories.value_counts(normalize=True))
sentence_len_categories.value_counts(normalize=True).sort_index().plot(kind="bar")
plt.show()

# 토큰 단위 지문 길이 출력(시간이 1분정도 걸림)
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
print()
print("토큰 단위 문장 길이(95 percentile) =",\
      train_data[args.column].apply(lambda x: len(tokenizer(x)['input_ids'])).agg([percentile(0.95)]).values[0])

train_data['token_len'] = train_data[args.column].apply(lambda x: len(tokenizer(x)['input_ids']))
print(train_data.agg({'token_len':['min', 'max', 'mean']}))

# 지문에 들어있는 특수문자 종류
not_en_ko_num_pattern = re.compile('[^ㄱ-ㅎ가-힣a-zA-Z0-9\s]')
known_special_chars_dict = defaultdict(int)
unk_special_chars_dict = defaultdict(int)

for sentence in train_data[args.column]:
    special_chars = not_en_ko_num_pattern.findall(sentence)
    if len(special_chars) != 0 :
        for char in special_chars:
            if tokenizer.tokenize(char) != ['[UNK]']:
                known_special_chars_dict[char] += 1
            else:
                unk_special_chars_dict[char] += 1

print()
print("klue/bert-base가 알고 있는 특수문자 개수 =", len(known_special_chars_dict))
# print(known_special_chars_dict)
print("klue/bert-base가 모르는 특수문자 개수 =", len(unk_special_chars_dict))
# print(unk_special_chars_dict)

print(f'***************************************************')