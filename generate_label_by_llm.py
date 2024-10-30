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
[|user|] 다음 뉴스의 헤드라인을 보고 기사가 어떤 주제인제 말해줘. 주제는 정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠가 있어. 답변은 간결하고 단답으로 해.
{}

예시 1:
헤드라인: 황총리 각 부처 비상대비태세 철저히 강구해야
토픽: 정치

예시 2:
헤드라인: 금융시장 충격 일단 소강국면…주가 낙폭 줄고 환율도 하락
토픽: 경제

예시 3:
헤드라인: 월미도 새 모노레일 내년 추석엔 달릴 수 있을까
토픽: 사회

예시 4:
헤드라인: 벚꽃 와인 마시며 봄 즐기세요
토픽: 생활문화

예시 5:
헤드라인: 트럼프 한국 등 방위비분담금 더 내라…양방향 도로 돼야
토픽: 세계

예시 6:
헤드라인: 네이버 모바일 연예판에도 AI 콘텐츠 추천 시스템 적용
토픽: IT과학

예시 7:
헤드라인: 대한항공 우리카드 꺾고 3연승…GS칼텍스 1라운드 전승종합
토픽: 스포츠

예시 8:
헤드라인: 中공안 장쑤 화학공단 폭발참사 책임자들 구금
토픽: 사회

예시 9:
헤드라인: 고3 확진자 나왔는데…부산서도 조마조마 3차 등교수업종합
토픽: 사회

예시 10:
헤드라인: 제주 4·3 아픔 보듬는 시집…꽃보다 먼저 다녀간 이름들
토픽: 사회

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
    
    # few_text = ["황총리 각 부처 비상대비태세 철저히 강구해야", "금융시장 충격 일단 소강국면…주가 낙폭 줄고 환율도 하락", "월미도 새 모노레일 내년 추석엔 달릴 수 있을까", "벚꽃 와인 마시며 봄 즐기세요", "트럼프 한국 등 방위비분담금 더 내라…양방향 도로 돼야", "네이버 모바일 연예판에도 AI 콘텐츠 추천 시스템 적용", "대한항공 우리카드 꺾고 3연승…GS칼텍스 1라운드 전승종합"]
    # few_topic = ["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"]
    # few_shop_promt = ''
    # for text, topic in zip(few_text, few_topic):
    #     few_shop_promt = few_shop_promt + EXAONE_FEW_SHOT_TEMP.format(text, topic)
    # logger.info(f'Few Shot Prompt : {few_shop_promt}')
    
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
                max_new_tokens=64
            )
            pred = tokenizer.decode(output[0])
            pred = pred.split("[|assistant|]")[-1]
            logger.info(f"Generate text : {pred}")
            lst.append(pred)
        except:
            lst.append("")
            logger.info(f"Error 발생")
            
    df['generate'] = lst
    logger.info(f"Save generate data : {df}")
    
    df.to_csv("./resources/processed/train_llm2.csv", encoding='utf-8-sig')
    

if __name__ == "__main__":
    main()