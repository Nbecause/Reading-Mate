'''
    데이터셋의 형태를 문학/비문학 분류 파인튜닝에 맞게 바꿔주는 코드
'''
from datasets import load_dataset
import argparse
from tqdm import tqdm
from utils import find_sentence_at_index, change_column_name

# 옵션 arg 받기
parser = argparse.ArgumentParser()
parser.add_argument('--csvout', help='추출할 csv 파일 이름(확장자 빼고)')
parser.add_argument('--subset', default='train', help='train, test, 혹은 valid ')
parser.add_argument('--dataset', default='squad_kor_v1', help='로컬/허깅페이스에서 가져올 데이터셋 이름')
parser.add_argument('--genre', default='non-lit', help='문학작품이면 literature, 비문학이면 non-lit')
args = parser.parse_args()

# 로컬에서 데이터셋 가져오기
dataset = load_dataset("csv", data_files=args.dataset)

# 판다스 데이터프레임으로 변환. 로컬 파일인 경우 args.subset은 train으로 해두면 된다.
train_data = dataset[args.subset].to_pandas()

# 칼럼 이름 바꾸거나 생성
col_name_dict = {'context':'context', 'label' : 'label'}
train_data = change_column_name(col_name_dict, train_data)

# 문학 데이터셋일땐 literature, 아닐땐 non-lit
train_data['label'] = args.genre

# 지문과 장르 이외의 불필요한 열 삭제
train_data = train_data[['context', 'label']]

# csv 파일로 추출
if args.csvout is not None:
    train_data.to_csv(f'./{args.csvout}.csv')
else:
    print("출력할 csv 파일 이름을 넣어주지 않아서 아무것도 출력되지 않습니다")