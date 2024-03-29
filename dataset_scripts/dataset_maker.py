'''
    ChatGPT를 이용해서 지문-질문 데이터셋을 만들어주는 코드
'''
import pandas as pd
from openai import AsyncOpenAI
from openai import OpenAI
import asyncio
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(
  api_key= 'OpenAI에서 카드 등록을 하고 api key를 받아서 넣으시면 됩니다'
)
filename = 'train_dataset_01'
n_loop = 20 # 리스트의 각 작품마다 n_loop개씩 지문-질문 데이터를 만들어줌. 약 150개를 넘어가면 OpenAI 서버에서 분당 토큰수 제한을 초과했다고 나온다.
title_list = ["유명 문학 작품", "한국 문학 작품", "아동 문학 작품", "장르 문학 작품", "고전 문학 작품"]

async def gptapi_async(prompt):
    completion = await client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return completion

# 비동기적으로 chatgpt 출력결과를 row(딕셔너리) 형태로 반환하는 함수
async def add_new_row_async(title):
    chatgpt_response_context = await gptapi_async(title + "의 문단 하나를 출력해 줘. 다섯 문장에서 열문장 사이로.")
    response_content_context = chatgpt_response_context.choices[0].message.content
    context_sentences = response_content_context.split('\"')
    # print(context_sentences)
    if len(context_sentences) == 1:
        pure_context = context_sentences[0]
    else:
        pure_context = context_sentences[1] # 보통 "공백" "내용" "출처와 작가" 순서임.

    chatgpt_response_question = await gptapi_async(pure_context+"\n이 지문에서 왜로 시작하는 질문을 하나만 생성해  줘")
    response_content_question = chatgpt_response_question.choices[0].message.content
    question_sentences = response_content_question.split('\"')
    # print(question_sentences)
    pure_question = question_sentences[-1]

    new_row = {'context': pure_context, 'question':pure_question}
    return new_row

# 비동기 함수를 여러번 실행하고 결과를 리스트로 반환하는 함수
async def process_async_tasks():
    tasks = []
    for i in range(n_loop):
        for title in title_list:
            tasks.append(add_new_row_async(title))

    return await tqdm_asyncio.gather(*tasks)

async def main():
    rows = await process_async_tasks()
    train_data = pd.DataFrame(rows) # row 리스트를 DataFrame화 한다.
    train_data.to_csv(f'./{filename}.csv')

if __name__ == "__main__":
    asyncio.run(main())