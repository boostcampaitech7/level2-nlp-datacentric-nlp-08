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
# [|user|] 다음 뉴스의 헤드라인을 보고 기사가 어떤 주제인제 말해줘. 주제는 정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠가 있어. 답변은 단답으로 해줘.
# {}
# [|assistant|]{}[|endofturn|]'''

# EXAONE_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
# [|user|] 다음 뉴스의 헤드라인을 보고 기사가 어떤 주제인제 말해줘. 주제는 정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠가 있어. 답변은 단답으로 해줘.
# {}
# [|assistant|]'''

EXAONE_FEW_SHOT_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
[|user|] 다음 뉴스의 헤드라인을 보고 어떤 주제인제 말해줘. 주제는 [정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠] 중에 있어. 
가능한 모든 주제를 3개까지 확률이 높은 순으로 나열해서 줘. 
설명은 간결하게 해.

### 분석포맷
1.주제: 생활문화 -> 키워드: "이유"

### 예시 1:
헤드라인: 김정은 "자주통일 새시대 열어나가야"
1.주제: 정치 -> 키워드: "김정은", "자주통일", "새시대"
2.주제: 세계 -> 키워드: "김정은", "자주통일", "새시대"
3.주제: 사회 -> 키워드: "자주통일", "새시대"

### 예시 2:
헤드라인: 사채 5조 9000억원...올 대비 55% 증가
1.주제: 경제 -> 키워드: "사채", "5조 9000억원", "55% 증가"

### 예시 3:
헤드라인: 모멘트 학교 급식 칸막이 설치 논란
1.주제: 사회 -> 키워드: "학교 급식 칸막이 설치"
2.주제: 생활문화 -> 키워드: "학교 급식"

### 예시 4:
헤드라인: 베트남, 맑은 날이 많은 나라, 관광 산업 발전 가능성 높아
1.주제: 생활문화 -> 키워드: "관광 산업 발전 가능성"
2.주제: 세계 -> 키워드: "베트남", "맑은 날"

### 예시 5:
헤드라인: 자베스 여, 정치권에 타협 촉구
1.주제: 세계 -> 키워드: "자베스 여"
2.주제: 정치 -> 키워드: "정치권", 타협"

### 예시 6:
헤드라인: 아프리카TV, F사와 시장 공략 위한 엔터테인먼트 회사와 MOU
1.주제: IT과학 -> 키워드: "엔터테인먼트", "MOU", "아프리카TV"
2.주제: 생활문화 -> 키워드: "엔터테인먼트", "시장 공략"

### 예시 7:
헤드라인: 월드컵 천재 사령탑 시코르 감독 한국전 승리...
1.주제: 스포츠 -> 키워드: "시시코르", "감독", "승리"
2.주제: 세계 -> 키워드: "월드컵", "한국전"

### 이제 다음 헤드라인이 어떤 주제인제 말해줘
헤드라인: {}
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
    
    df = pd.read_csv("./resources/processed/train_not_noise.csv")
    logger.info(f"Train dataset size: {len(df)}")
    
    lst = []
    for _, row in df.iterrows():
        # text = row['text']
        re_text = row['re_text'].strip()
        logger.info(f"text : {re_text}")
        try:
            text_prompt = EXAONE_FEW_SHOT_TEMP.format(re_text)
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
            
    df['gen_topic'] = lst
    logger.info(f"Save generate data : {df}")
    
    df.to_csv("./resources/processed/train_not_noise_topic.csv", encoding='utf-8-sig', index=0)
    

if __name__ == "__main__":
    main()