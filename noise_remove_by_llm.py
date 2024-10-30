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
[|user|] 다음 뉴스의 헤드라인을 보고 맞춤법을 교정해줘. 답변은 간결하고 단답으로 해. 설명은 절대 하지마.
{}
[|assistant|]{}[|endofturn|]'''

EXAONE_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
[|user|] 다음 뉴스의 헤드라인을 보고 맞춤법을 교정해줘. 답변은 간결하고 단답으로 해. 설명은 절대 하지마.
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
    
    few_text = ["NH투자 1월 옵션 만기일 매도 우세", "정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보", "K찰.국DLwo 로L3한N% 회장 2 T0&}송="
            "m 김정) 자주통일 새,?r열1나가야1보", "pI美대선I앞두고 R2fr단 발] $비해 감시 강화", "프로야구~롯TKIAs광주 경기 y천취소"]
    few_retext = ["NH투자 1월 옵션 만기일 매도 우세", "정부, '주파수 미사용' KT에 이용기간 2년 단축 처분(종합2보)","경찰, '국회 불법 로비' 한어총 회장 등 20명 송치"
                "김정은 \"자주통일 새시대 열어나가야\"(2보)", "軍 \"美대선 앞두고 北 무수단 발사 대비해 감시 강화\"", "프로야구 롯데-KIA 광주 경기 우천취소"]
    few_shop_promt = ''
    for text, topic in zip(few_text, few_retext):
        few_shop_promt = few_shop_promt + EXAONE_FEW_SHOT_TEMP.format(text, topic)
    logger.info(f'Few Shot Prompt : {few_shop_promt}')
    
    df = pd.read_csv("./resources/raw_data/train.csv")
    logger.info(f"Train dataset size: {len(df)}")
    
    lst = []
    for _, row in df.iterrows():
        text = row['text']
        logger.info(f"Origin text : {text}")
        try:
            text_prompt = EXAONE_TEMP.format(text)
            input_ids = tokenizer(few_shop_promt + text_prompt, return_tensors="pt")['input_ids']
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
    
    df.to_csv("./resources/processed/train_noise_remove_llm.csv", encoding='utf-8-sig')
    

if __name__ == "__main__":
    main()