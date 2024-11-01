import torch
import random
import logging
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login()

logger = logging.getLogger("llm")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

# EXAONE_FEW_SHOT_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
# [|user|] 다음 뉴스의 헤드라인을 보고 맞춤법을 교정해줘. 답변은 간결하고 단답으로 해. 설명은 절대 하지마.
# {}
# [|assistant|]{}[|endofturn|]'''

# EXAONE_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
# [|user|] 다음 뉴스의 헤드라인을 보고 맞춤법을 교정해줘. 답변은 간결하고 단답으로 해. 설명은 절대 하지마.
# {}
# [|assistant|]'''

EXAONE_FEW_SHOT_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
[|user|] 다음 문장에서 특수문자, 숫자, 영어로 대체된 부분을 올바른 단어로 복원해. 복원할게 없으면 원본 그대로 답변해. 복원된 문장을 보고 어떤 주제의 기사인지 말해줘. 주제는 정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠 중에 있어

설명은 간결하게 해.

예시 1:
원본: NH투자 1월 옵션 만기일 매도 우세
복원: NH투자 1월 옵션 만기일 매도 우세
주제: 경제

예시 2: 
원본: 정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보
복원: 정부, '주파수 미사용' KT에 이용기간 2년 단축 처분(종합2보)
주제: IT과학

예시 3:
원본: K찰.국DLwo 로L3한N% 회장 2 T0&송=
복원: 경찰, '국회 불법 로비' 한어총 회장 등 20명 송치
주제: 정치

예시 4:
원본: m 김정) 자주통일 새,?r열1나가야1보
복원: 김정은 "자주통일 새시대 열어나가야"(2보)
주제: 정치

예시 5:
원본: pI美대선I앞두고 R2fr단 발] $비해 감시 강화
복원: 軍 "美대선 앞두고 北 무수단 발사 대비해 감시 강화"
주제: 세계

예시 6:
원본: 프로야구~롯TKIAs광주 경기 y천취소
복원: 프로야구 롯데-KIA 광주 경기 우천취소
주제: 스포츠

예시 7:
원본: 버닝썬 게이트 다룬 SBS 그것이 알고 싶다 11.2％
복원: 버닝썬 게이트 다룬 SBS 그것이 알고 싶다 11.2％
주제: 사회

예시 8:
원본: 보령소식 보령시 시간선택제 공무원 3명 모집
복원: 보령소식 보령시 시간선택제 공무원 3명 모집
주제: 생활문화

예시 9:
원본: 안보리 대북결의안 2270호 이행보고서 제출한 나라 70개 육박
복원: 안보리 대북결의안 2270호 이행보고서 제출한 나라 70개 육박
주제: 세계

예시 10:
원본: 고3 확진자 나왔는데 부산서도 조마조마 3차 등교수업종합
복원: 고3 확진자 나왔는데 부산서도 조마조마 3차 등교수업종합
주제: 사회

이제 다음 문장을 복원해주고 주제를 말해줘.
원본: {}

[|assistant|]'''


def main():
    logger.info('*** START ***')

    model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
    logger.info(f'Model & Tokenizer : LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct')
    
    df = pd.read_csv("./resources/raw_data/train.csv")
    logger.info(f"Train dataset size: {len(df)}")
    
    lst = []
    for _, row in df.iterrows():
        text = row['text']
        logger.info(f"Origin text : {text}")
        
        try:
            text_prompt = EXAONE_FEW_SHOT_TEMP.format(text)
            input_ids = tokenizer(text_prompt, return_tensors="pt")['input_ids']
            output = model.generate(
                input_ids.to("cuda"),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=128
            )
            pred = tokenizer.decode(output[0])
            pred = pred.split("[|assistant|]")[-1]
            logger.info(f"Generate text : {pred}")
            lst.append(pred)
        except:
            lst.append("")
            logger.info(f"Error 발생")
            
    df['re_text'] = lst
    logger.info(f"Save generate data : {df}")
    
    df.to_csv("./resources/processed/v3/train_noise_remove_llm3.csv",encoding='utf-8-sig')
    

if __name__ == "__main__":
    main()