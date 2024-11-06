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
[|user|] 너는 뉴스 헤드라인을 분석하여 주제를 예측하는 AI 모델이야. 주어진 헤드라인을 분석하고, 가장 적합한 주제 번호를 0부터 6까지의 정수로 제시해줘.
 
### 각 주제 번호에 해당하는 헤드라인은 다음과 같아.
주제 0 예시:
헤드라인 1: 도자기 작품 감상하는 손님들
헤드라인 2: 여행의 매력, 제주에서 다양한 취향을 만족시키다...문화 축제
헤드라인 3: 공지 1년 만의 소설 출간, 할머니는 죽지 않는다
헤드라인 4: 한국 코믹콘 축제 개막...다양한 콘텐츠 선보여
헤드라인 5: 숲속의 숲 해설센터, 청소년 교육 프로그램 운영

주제 1 예시:
헤드라인 1: 프로농구 선수 윤호영, 무릎 수술 비용 650만원 지원
헤드라인 2: GD 감독, 2로축구연맹 월드컵 7개국 순회 경기 개최
헤드라인 3: 박항서 매직 베트남 축구 대표팀
헤드라인 4: MLB 로키스, 피츠버그와 경기! 뉴에이지 메달리스트 감독 취임
헤드라인 5: 삼성이 극적인 역전을... 손혁 감독, 연패 늪에서 벗어나

주제 2 예시:
헤드라인 1: 박원주 공정거래위원회 위원장, 누리과정 문제 해결 위해 소통하자는데 반응 주목
헤드라인 2: 문재인 정부, 민간 2동 7사위 철거 명령
헤드라인 3: 선거 여론조사 결과 유출 논란
헤드라인 4: 친박 원내대표 표오도원, 유승민과 결별...단일화 실패(종합)
헤드라인 5: 남북정상회담 준비가 생명, 남북 리허설 또 리허설

주제 3 예시:
헤드라인 1: 모멘트 학교 급식 칸막이 설치 논란
헤드라인 2: 언론인, 조부모, 장애인...시각장애인 폭 넓게 포용하는 시인
헤드라인 3: 강원 전교조 교사들 원스트라이크 아웃 제도 도입 반발
헤드라인 4: 코로나19 환자 36일째 단 한 명도 입원하지 않아 건강 악화, 병원들 비상 (종합)
헤드라인 5: 남도 지역 언론 5곳 포함 보도 요청

주제 4 예시:
헤드라인 1: 정부, '주파수 미사용' KT에 이용기간 2년 단축 처분(종합2보)
헤드라인 2: 스마트폰 '블루라이트' 막는 필름 출시
헤드라인 3: 사티아 나델라 인터랙티브 신제품 발표 Y투데이 인터뷰해 미래 먹거리 논의
헤드라인 4: 갤럭시 S9 퀀텀 패키지 출시, 5천대 한정 판매
헤드라인 5: 구글 AI 진단은 의사가 할 수 없다...X단 보조 역할

주제 5 예시:
헤드라인 1: 오일 매력적인 모델들이 열을 올리며 약 3억 주가 고평가되고 있다
헤드라인 2: 삼성전자, 3분기 영업이익 6조원 감소
헤드라인 3: 최희림 SK텔레콤 헬릭스비전 병의원 서비스보다 연매출 70% 증가(종합)
헤드라인 4: 증시 신제품 출시, 펀드 수익률 상승
헤드라인 5: 비트코인 거래, 올해 50% 이상 증가

주제 6 예시:
헤드라인 1: 美 MBA 여성 비율 계속 증가, 올해 입학생 중 40%가 여성
헤드라인 2: 프랑스 마크롱 대통령 G7 정상회의 참석
헤드라인 3: 우크라이나, 러시아에 맞서 '전술핵 맞설 무기' 개발 추진
헤드라인 4: 사우디 왕세자와 미국 방문...유시엔조 지속 합의종합
헤드라인 5: 시진핑 북한 방문 민생 행보 IT 업계까지 '만반 준비'

### 헤드라인을 분석할 때 다음 특징들을 고려해:
- 키워드: 헤드라인에 사용된 주요 단어나 표현이 특정 주제와 연관되어 있는지 확인해.
- 인물 및 기관: 언급된 인물이나 기관이 특정 분야와 관련이 있는지 파악해.
- 사건 및 활동: 헤드라인이 설명하는 사건이나 활동이 어떤 분야에 속하는지 판단해.
- 맥락: 전반적인 맥락이 어떤 주제 영역에 가장 적합한지 고려해.

### 분석 결과를 다음과 같은 형식으로 제시해줘. 가능한 모든 주제를 최대 3개까지 확률이 높은 순으로 나열해서 줘. 
주제 번호: [0-6 중 하나의 정수] -> 키워드: [헤드라인에서 해당 주제로 판단한 근거를 키워드로 설명]

예를들어, "아프리카TV, F사와 시장 공략 위한 엔터테인먼트 회사와 MOU"라는 헤드라인이 주어졌다면 다음과 같이 분석할 수 있어.
1. 주제 번호: 4 -> 키워드 : "엔터테인먼트", "MOU", "아프리카TV"
2. 주제 번호: 0 -> 키워드: "엔터테인먼트", "시장 공략"

### 이제 주어진 헤드라인을 분석하여 가장 적합한 주제를 예측해줘.
{}
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
    
    df = pd.read_csv("./resources/processed/train_noisy.csv")
    logger.info(f"Train dataset size: {len(df)}")
    
    lst = []
    for _, row in df.iterrows():
        re_text = row['re_text'].strip()
        logger.info(f"text : {re_text}")
        try:
            text_prompt = EXAONE_FEW_SHOT_TEMP.format(re_text)
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
            
    df['gen_topic'] = lst
    logger.info(f"Save generate data : {df}")
    
    df.to_csv("./resources/processed/train_noisy_generation_topic.csv", encoding='utf-8-sig', index=0)
    

if __name__ == "__main__":
    main()