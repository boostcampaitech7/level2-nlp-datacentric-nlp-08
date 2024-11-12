import torch
import random
import logging
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

hf_token = "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"
login(hf_token)

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

logger = logging.getLogger("llm")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# You are EXAONE model from LG AI Research, a helpful assistant. 
# 너는 자극적인 헤드라인으로 이목을 끄는데 능통한
# 연령70대를 위한 뉴스를 적는 편집자야
EXAONE_FEW_SHOT_TEMP = '''[|system|] 당신은 EXAONE이라는 AI 언어 모델입니다. 당신의 역할은 주어진 텍스트와 주제 번호를 바탕으로 아래 지침에 따라 기사 헤드라인을 작성하는 것입니다.
1. 아래 주어진 주제 번호와 키워드 정보를 숙지하세요.
2. 주어진 텍스트와 주제 번호를 보고 주제 번호의 키워드 및 특징과 일치하는 뉴스 기사 헤드라인을 생성하세요.
3. 헤드라인 생성 시 다음 규칙을 준수하세요:
    - 주어진 텍스트의 사건이나 상황과 전혀 다른 내용을 다루세요. 또는 완전히 다른 관점의 내용을 다루세요.
    - 주어진 텍스트에 있는 고유명사를 절대 사용하지 마세요. 완전히 다른 인물, 팀, 기업 등을 사용하세요.
    - 현실적이고 그럴듯한 헤드라인을 작성하세요.
    - 헤드라인만 출력하고 생성 텍스트에 대한 부가 설명은 하지 마세요.
    - 여러 주제 번호에 걸치는 헤드라인은 생성하지 마세요.
4. 생성된 헤드라인을 다음 형식으로 출력하세요.
    1.헤드라인: [생성된 헤드라인]
    2.헤드라인: [생성된 헤드라인]
이 지침을 따라 새로운 뉴스 기사 제목을 생성하세요.

## 주제 번호 및 키워드
### 주제 번호 0:
키워드: 책/문학, 만화/웹툰, 종교, 공연/전시, 학술/문화재, 미디어, 여행/레저, 생활, 건강정보, 자동차/시승기, 도로/교통, 음식/맛집, 패션/뷰티, 날씨, 생활 일반, 문화 일반 
특징:
- 가벼운 주제의 헤드라인 기사
- 여가생활이나, 공모전과 같은 기사

### 주제 번호 1:
키워드: 야구, 축구, 농구/배구, 골프, 해외야구, 해외축구, N골프, e스포츠 , 수영, 태권도
특징:
- 선수 이름이 나오는 기사
- 선수의 부상재활이나, 이적, 팀의 성과 등을 포함하는 기사

### 주제 번호 2:
키워드: 대통령실/총리실, 국회/정당, 외교, 국방, 북한, 행정, 탈북
특징:
- 한국외의 국가가 헤드라인에 포함된다면, 외교관련 문제만 포함하는 기사
- 정부기관의 행정에 관한 기사
- 북한과 한국 사이에서의 일에 관련된 기사

### 주제 번호 3:
키워드: 사건/사고, 사망, 추락사, 실형, 전수조사, 활동 정지, 징계, 법원/검찰, 교육, 복지/노동, 환경, 여성/아동, 재외동포, 다문화, 인권/복지, 식품/의료, 지역
특징:
- 사건 사고를 다루는 기사
- 공적인 내용을 다루는 기사
- 스포츠를 제외한 기사

### 주제 번호 4:
키워드: 모바일, 인터넷/SNS, 통신/뉴미디아, IT일반, 보안/해킹, 컴퓨터, 게임/리뷰, 해커, 해킹, 암호(암호화폐), 소행성, (유전자 등의) 발견, 랜섬웨어, 뇌
특징:
- 새로운 기술에 대해서 다루는 기사
- 어떤 새로운 것들을 출시했다는 내용이 담긴 기사

### 주제 번호 5:
경제/정책, 금융, 부동산, 취업/창업, 소비자, 산업/재계, 중기/벤처, 글로벌 경제, 생활 경제
특징:
- 실질적인 퍼센트(%)가 포함된 기사
- 증가, 감소, 올랐다, 내렸다 등의 증감표현이 있는 기사

### 주제 번호 6:
키워드: 특파원, 미국/북미, 중국, 일본, 아시아/호주, 유럽, 중남미, 중동/아프리카, 국제기구, 총격, 로커비 사건
특징:
- 한국과 다른 나라간의 외교관련된 것이 **아닌** 다른 나라끼리의 소식과 관련된 기사
- 북한과 한국이 아닌 다른 나라 사이에서의 일에 관련된 기사
[|endofturn|]

[|user|]
다음 텍스트와 주제 번호를 보고 동일한 주제의 뉴스 기사 제목을 5개 생성해줘. 대신 완전히 다른 내용으로 생성해줘. 뒤에 부가 설명은 절대 덧붙이지 마. 경고 했다. 제대로 못하면 다른 모델로 교체할거야.
주제 번호 : {}
텍스트: {}
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
    
    df = pd.read_csv("kdh/hard_to_predict.csv")
    logger.info(f"Train dataset size: {len(df)}")
    
    lst = []
    # p_dic = {0:p0, 1:p1, 2:p2, 3:p3, 4:p4, 5:p5, 6:p6}
    for _, row in df.iterrows():
        target = row['target']
        # p = p_dic[int(target)]
        target_name = row['target_name']
        text = row['text'].strip()
        logger.info(f"topic : {target}, target_name : {target_name}, text : {text},")
        # p, target, 
        try:
            text_prompt = EXAONE_FEW_SHOT_TEMP.format(target, text)
            input_ids = tokenizer(text_prompt, return_tensors="pt")['input_ids']
            output = model.generate(
                input_ids.to("cuda"),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                #do_sample=True,
                #temperature=0.8,
                # top_=1.2,
            )
            pred = tokenizer.decode(output[0])
            pred = pred.split("[|assistant|]")[-1]
            
            logger.info(f"Generate text : {pred}")
            pred = [sen.split(':')[-1].strip().replace("[|endofturn|]", "") for sen in pred.split('\n') if sen != "" and sen[0].isdigit()]
            
            for sen in pred:
                copy_row = row.copy()
                copy_row['text'] = sen
                lst.append(copy_row)
        except:
            lst.append("")
            logger.info(f"Error 발생")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    lst = pd.DataFrame(lst)
    new_df = lst
    new_df = pd.concat([df, lst], axis=0, ignore_index=True).sort_values(by="ID").drop_duplicates().reset_index(drop=True)
    logger.info(f"Save generate data : {df.head()}")
    
    new_df.to_csv("kdh/hard_case_generation.csv", encoding='utf-8-sig', index=0)
    

if __name__ == "__main__":
    main()