from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import os

repo = "davidkim205/iris-7b"
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(repo)


def generate(prompt):
    encoding = tokenizer(
        prompt,
        return_tensors='pt',
        return_token_type_ids=False
    ).to("cuda")
    gen_tokens = model.generate(
        **encoding,
        max_new_tokens=2048,
        temperature=1.0,
        num_beams=5,
    )
    prompt_end_size = encoding.input_ids.shape[1]
    result = tokenizer.decode(gen_tokens[0, prompt_end_size:])

    result=result.replace("</s>","")
    return result

def translate_ko2en(text):
    prompt = f"[INST] 다음 문장을 영어로 번역하세요.{text} [/INST]"
    return generate(prompt)

def translate_en2ko(text):
    prompt = f"[INST] 다음 문장을 한글로 번역하세요.{text} [/INST]"
    return generate(prompt)



def backtranslate(text):
    ko_2_en=translate_ko2en(text)
    en_2_ko=translate_en2ko(ko_2_en)

    return en_2_ko

# 데이터 로드 및 저장 파일 설정


input_file = "noisy_headline_filtered.csv"
output_file = "not_noisy_only_backtranslated_output.csv"

# 중간 결과 파일이 있으면 로드해서 이어서 진행
if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file)
    start_index = len(processed_df)
    print(f"{start_index}번째 행부터 다시 시작합니다.")
else:
    processed_df = pd.DataFrame(columns=["text", "back_translate_result"])
    start_index = 0

filtered_df = pd.read_csv(input_file)

# 백트랜슬레이션 적용 및 오류 발생 시 저장
for idx in tqdm(range(start_index, len(filtered_df)), initial=start_index, total=len(filtered_df)):
    text = filtered_df.loc[idx, "text"]
    
    try:
        result = backtranslate(text)
    except Exception as e:
        print(f"{idx}번째 행에서 오류 발생: {e}")
        processed_df.to_csv(output_file, index=False)
        break
    
    # 결과를 데이터프레임에 추가
    processed_df = pd.concat([processed_df, pd.DataFrame({"text": [text], "back_translate_result": [result]})], ignore_index=True)

# 모든 행 처리 완료 후 저장.
processed_df.to_csv(output_file, index=False)