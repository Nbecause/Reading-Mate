def find_sentence_at_index(text, target_index):
    '''여러 문장들이 있는 스트링에서 특정 인덱스를 받으면 해당 인덱스를 포함하는 문장을 리턴하는 함수'''
    sentences = text.split('.')  # 마침표를 기준으로 문장을 분리합니다.
    current_index = 0
    for sentence in sentences:
        sentence_length = len(sentence) + 1  # 현재 문장의 길이와 마침표(1)를 더합니다.
        if target_index < current_index + sentence_length:
            start_index = current_index
            end_index = current_index + sentence_length - 1
            return sentence.strip(), start_index, end_index
        current_index += sentence_length
    return None, None, None


def change_column_name(col_name_dict, train_data):
    '''칼럼 이름 바꾸거나 생성
    Args: 
        train_data: pd.dataframe을 받습니다.
        col_name_dict: 원래 칼럼 이름과 바꿀 칼럼 이름이 key-value로 짝지어진 딕셔너리를 받습니다.
    Returns:
        수정된 train_data를 반환합니다.
    '''
    for key, val in col_name_dict.items():
        # key 칼럼이 존재하면 이름을 val로 바꿈
        if key in train_data.columns:
            train_data = train_data.rename(columns={key: val})
        # key 칼럼이 존재하지 않으면 새 val 칼럼을 만듬
        else:
            train_data[val] = None
    return train_data