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
EXAONE_FEW_SHOT_TEMP = '''[|system|] 너는 숙련된 뉴스 편집자야. 아래의 조건들을 바탕으로 적절한 뉴스 헤드라인을 작성해야 해. 
## 아래의 조건은 필수적으로 지켜야해.
1. 주어진 주제 번호에 해당하는 키워드와 특징을 숙지해야 해.
2. 주제 번호에 관해서 너가 알고있는 사람 이름이나, 기업 이름 등의 고유명사를 사용해도 돼.
3. 설명은 출력하지 않아도 돼. 반드시 헤드라인만 출력 해줘.
4. 주제 번호는 모두 독립적이야. 여러 개의 주제번호에 포함되는 헤드라인은 절대 생성해서는 안돼.[|endofturn|]
[|user|] 

```
## 주어진 주제 번호
### 주제 번호 0:
키워드: 책/문학, 만화/웹툰, 종교, 공연/전시, 학술/문화재, 미디어, 여행/레저, 생활, 건강정보, 자동차/시승기, 도로/교통, 음식/맛집, 패션/뷰티, 날씨, 생활 일반, 문화 일반 
특징:
- 가벼운 주제의 헤드라인 기사
- 여가생활이나, 공모전과 같은 기사

### 주제 번호 1:
키워드: 야구, 축구, 농구/배구, 골프, 해외야구, 해외축구, N골프, e스포츠 , 수영, 태권도, 
특징:
- 선수 이름이 나오는 기사
- 선수의 부상재활이나, 이적, 팀의 성과 등을 포함하는 기사

### 주제 번호 2:
키워드: 대통령실/총리실, 국회/정당, 외교, 국방, 북한, 행정
특징:
- 한국외의 국가가 헤드라인에 포함된다면, 외교관련 문제만 포함하는 기사
- 정부기관의 행정에 관한 기사
- 북한과 한국 사이에서의 일에 관련된 기사

### 주제 번호 3:
키워드: 사건/사고, 법원/검찰, 교육, 복지/노동, 환경, 여성/아동, 재외동포, 다문화, 인권/복지, 식품/의료, 지역
특징:
- 사건 사고를 다루는 기사
- 공적인 내용을 다루는 기사
- 스포츠를 제외한 기사

### 주제 번호 4:
키워드: 모바일, 인터넷/SNS, 통신/뉴미디아, IT일반, 보안/해킹, 컴퓨터, 게임/리뷰
특징:
- 새로운 기술에 대해서 다루는 기사
- 어떤 새로운 것들을 출시했다는 내용이 담긴 기사

### 주제 번호 5:
경제/정책, 금융, 부동산, 취업/창업, 소비자, 산업/재계, 중기/벤처, 글로벌 경제, 생활 경제
특징:
- 실질적인 퍼센트(%)가 포함된 기사
- 증가, 감소, 올랐다, 내렸다 등의 증감표현이 있는 기사

### 주제 번호 6:
키워드: 특파원, 미국/북미, 중국, 일본, 아시아/호주, 유럽, 중남미, 중동/아프리카, 국제기구
특징:
- 한국과 다른 나라간의 외교관련된 것이 **아닌** 다른 나라끼리의 소식과 관련된 기사
- 북한과 한국이 아닌 다른 나라 사이에서의 일에 관련된 기사
```

### 너가 작성할 헤드라인은 다음과 같은 형식을 따라야 해.
1.헤드라인: [주제에 맞춰 작성된 헤드라인]
2.헤드라인: [주제에 맞춰 작성된 헤드라인]

### 위에 주어진 주제와 각 주제의 키워드 정보를 바탕으로 아래 주어질 텍스트의 주제를 파악하고, 그와 같은 주제의 기사 헤드라인을 5개 작성해줘. 하지만 내용은 웬만하면 다른 내용이었으면 좋겠어. 제발 제대로 작성해줘. 부탁할게 제발. 이렇게 빌게. 한 번만 도와주라 진짜. LG 제품만 쓸게 앞으로.
주제 번호: {}
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
        # try:
        text_prompt = EXAONE_FEW_SHOT_TEMP.format(target, text)
        input_ids = tokenizer(text_prompt, return_tensors="pt")['input_ids']
        output = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=2056,
            # temperature=1.2,
            # top_=1.2,
        )
        pred = tokenizer.decode(output[0])
        pred = pred.split("[|assistant|]")[-1]
        
        logger.info(f"Generate text : {pred}")
        lst.append(pred)
        # except:
            # lst.append("")
            # logger.info(f"Error 발생")
    del model
    del tokenizer
    torch.cuda.empty_cache()
        
    df['gen_headline'] = lst
    logger.info(f"Save generate data : {df}")
    
    df.to_csv("kdh/hard_case_generation.csv", encoding='utf-8-sig', index=0)
    

if __name__ == "__main__":
    main()