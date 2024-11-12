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


EXAONE_FEW_SHOT_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
[|user|] 너는 숙련된 뉴스 편집자야. 주어진 노이즈가 삽입된 텍스트와 주제 번호를 바탕으로 적절한 뉴스 헤드라인을 작성해야 해.

### 다음 지침을 따라야해.
- 제시된 주제 번호에 해당하는 예시 헤드라인들의 내용을 통해 주제를 파악하세요.
- 노이즈가 삽입된 텍스트를 주의 깊게 읽으세요.
- 텍스트에서 의미 있는 단어나 구문을 식별하세요.
- 식별된 키워드와 주제에 맞는 새롭고 창의적인 헤드라인을 작성하세요.
- 키워드를 식별할 수 없다면 주제에 맞는 새롭고 창의적인 헤드라인을 만드세요.
- 생성된 헤드라인은 예시의 내용을 포함하지 않아야 합니다.
- 설명은 출력하지 마세요.

### 주제 번호 0 예시:
예시 1: 도자기 작품 감상하는 손님들
예시 2: 여행의 매력, 제주에서 다양한 취향을 만족시키다...문화 축제
예시 3: 공지 1년 만의 소설 출간, 할머니는 죽지 않는다
예시 4: 한국 코믹콘 축제 개막...다양한 콘텐츠 선보여
예시 5: 숲속의 숲 해설센터, 청소년 교육 프로그램 운영
예시 6: 인제 자작나무 숲 입산 통제 15일부터 2개 구간
예시 7: 6월 5일 날씨: 낮 최고 8도, 강릉 미세먼지 나쁨
예시 8: 세종시 금정문화권 2단계 개발 사업 7월 착수 문화제 개최
예시 9: 오늘도 여전히 정선·의성 9.1도, 서울 8도
예시 10: 스타필드 하남 개장 후 이틀 동안 3만 명 방문

### 주제 번호 1 예시:
예시 1: 프로농구 선수 윤호영, 무릎 수술 비용 650만원 지원
예시 2: GD 감독, 2로축구연맹 월드컵 7개국 순회 경기 개최
예시 3: 메시, 바르셀로나 떠나 빌바오 입단
예시 4: MLB 로키스, 피츠버그와 경기! 뉴에이지 메달리스트 감독 취임
예시 5: 삼성이 극적인 역전을... 손혁 감독, 연패 늪에서 벗어나
예시 6: 올림픽 상비군 대표팀, 베트남과 경기 8일 오후 7시 개최
예시 7: 아약스 맨시티 677골...28년 구단 최다수 득점자

### 주제 번호 2 예시:
예시 1: 박원주 공정거래위원회 위원장, 누리과정 문제 해결 위해 소통하자는데 반응 주목
예시 2: 문재인 정부, 민간 2동 7사위 철거 명령
예시 3: 선거 여론조사 결과 유출 논란
예시 4: 친박 원내대표 표오도원, 유승민과 결별...단일화 실패(종합)
예시 5: 남북정상회담 준비가 생명, 남북 리허설 또 리허설
예시 6: 총선 5개월 앞인데 선거운동 4정 안갯속...정치인들 발벗고 나서
예시 7: 트럼프/ 일본 방문; 북한과 무역적자 문제 논의 32
예시 8: 김부겸 전 총리 이달 안 결정...단일 후보 앞으로 나아가야
예시 9: 민주화운동기념법안 9일 난항...고민 중인 여당, 야당은 반대
예시 10: 시진핑 중국 대표와 인도 총리 회담

### 주제 번호 3 예시:
예시 1: 모멘트 학교 급식 칸막이 설치 논란
예시 2: 언론인, 조부모, 장애인...시각장애인 폭 넓게 포용하는 시인
예시 3: 강원 전교조 교사들 원스트라이크 아웃 제도 도입 반발
예시 4: 코로나19 환자 36일째 단 한 명도 입원하지 않아 건강 악화, 병원들 비상 (종합)
예시 5: 학교폭력 가해자 엄벌 요구하는 학생들
예시 6: 광주 북구, 청소년 계도 프로그램 시작
예시 7: 정부, '인터넷 중독' 청소년 100만 명 구조 요청
예시 8: 종교시설 방역수칙 위반 사업자 증가...교황청 지침 존재한
예시 9: 한국노총 주 4일제 일자리 논의, 주노동자들 의견 수렴 중
예시 10: 어린이 보호구역 안전 강화...단속카메라 확대 및 펫티켓 교육

### 주제 번호 4 예시:
예시 1: 정부, '주파수 미사용' KT에 이용기간 2년 단축 처분(종합2보)
예시 2: 스마트폰 '블루라이트' 막는 필름 출시
예시 3: 사티아 나델라 인터랙티브 신제품 발표 Y투데이 인터뷰해 미래 먹거리 논의
예시 4: 갤럭시 S9 퀀텀 패키지 출시, 5천대 한정 판매
예시 5: 구글 AI 진단은 의사가 할 수 없다...X단 보조 역할
예시 6: 혈관 속 움막 모더니즘 봇 2개...자율주행로보틱스 연구로
예시 7: 삼성전자, 새 버전 공개...AI로 업무 효율 높여

### 주제 번호 5 예시:
예시 1: LGU+ 통신요금 4만원 이상 고객에게 고금리 적금 출시
예시 2: 삼성전자, 3분기 영업이익 6조원 감소
예시 3: 최희림 SK텔레콤 헬릭스비전 병의원 서비스보다 연매출 70% 증가(종합)
예시 4: 증시 신제품 출시, 펀드 수익률 상승
예시 5: 비트코인 거래, 올해 50% 이상 증가
예시 6: 개인투자자들이 소액으로 주식에 투자하는 '개미'들이 시장에서 큰 역할을 하고 있다
예시 7: 지난달 분양 3천268채, 7개월 연속 증가세
예시 8: 포스텍, ICT 융합 로보틱스 스타트업에 600억 투자

### 주제 번호 6 예시:
예시 1: 美 MBA 여성 비율 계속 증가, 올해 입학생 중 40%가 여성
예시 2: 프랑스 마크롱 대통령 G7 정상회의 참석
예시 3: 우크라이나, 러시아에 맞서 '전술핵 맞설 무기' 개발 추진
예시 4: 사우디 왕세자와 미국 방문...유시엔조 지속 합의종합
예시 5: 軍 "美대선 앞두고 北 무수단 발사 대비해 감시 강화"
예시 6: 시진핑 북한 방문 민생 행보 IT 업계까지 '만반 준비'
예시 7: 이스라엘, 팔레스타인 가자지구에 로켓 발사 대응 태세 강화
예시 8: 미국 제재 유예 불허에 원유 수출 목 조르는 이란, 호르무즈 해협 봉쇄 경고
예시 9: 전 세계적으로 기후 변화에 대한 우려가 커지고 있어
예시 10: 이란 우라늄 원심분리기 가동...합의 이행소종합

### 뉴스 헤드라인을 다음과 같은 형식으로 최대 2개까지 제시해줘.
1.헤드라인: [주제에 맞춰 작성된 헤드라인]
2.헤드라인: [주제에 맞춰 작성된 헤드라인]

### 위의 정보를 바탕으로 새로운 뉴스 헤드라인을 작성해줘.
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
    
    df = pd.read_csv("./resources/processed/train_noisy.csv")
    logger.info(f"Train dataset size: {len(df)}")
    
    lst = []
    for _, row in df.iterrows():
        target = row['target']
        target_name = row['target_name']
        text = row['text'].strip()
        logger.info(f"topic : {target_name}, text : {text}")
        try:
            text_prompt = EXAONE_FEW_SHOT_TEMP.format(target, text)
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
            
    df['gen_headline'] = lst
    logger.info(f"Save generate data : {df}")
    
    df.to_csv("./resources/processed/train_noisy_generation_headline.csv", encoding='utf-8-sig', index=0)
    

if __name__ == "__main__":
    main()