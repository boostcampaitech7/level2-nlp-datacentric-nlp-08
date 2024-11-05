train_noisy.csv : 노아,사이먼 intersection 비교해서 보완한 데이터, re_text는 LLM으로 클렌징 진행한 텍스트.
train_noisy_headline.csv : 사이먼, 이전 train_noisy.csv에서 추가로 headline 생성한 데이터

**back translation ver 1.0**
eng_back_trans.csv : backtranslation result
aug_dataset.csv : backtranslation train dataset


**back translation ver 2.0**
noisy_headline_filtered.csv : backtranslation 전에 특수기호로 filtering한 결과값
backtranslated_output.csv : backtranslation result
backtranslate_ver2.csv : backtranslation train dataset

**back translation ver 3.0**
train_20241104.csv : 11,614개 의 병합된 데이터 (noisy_text : 10,401 + not_noisy_text : 1,213)
train_20241104_not_noisy_only.csv : train_20241104.csv에서 not_noisy만 선택한 데이터
not_noisy_only_backtranslated_output.csv : train_20241104_not_noisy_only.csv backtranslation result

**inference**
test.csv : inference시 input