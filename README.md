# SMART-TTS_Single_Emotional
Transformer-TTS 기반의 SMART-TTS의 Single speaker emotional TTS 모델입니다.
공개된 코드는 2020년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한
"소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발"
과제의 일환으로 공개된 코드입니다.

SMART-TTS_Single_Emotional 모델은 Transformer-TTS 모델을 기반으로
Reference로 부터 variable-length style embedding을 추출하여 반영하는 Unsupervised Style TTS 모델입니다.

Transformer-TTS 모델을 기반으로 하여 아래 부분들을 개선하였습니다.
* Monotonic interpolation을 이용한 Attention failure 개선
* Sliding window attention masking을 이용한 Attention failure 개선 및 stop token 대체
* Variable-length style embedding 을 이용한 global & local style embedding 추출
* Scaled dot-product attention을 이용한 style conditioning
* SMART-Vocoder를 이용한 고품질 합성음
* SMART-G2P를 이용한 한/영 code-mixed text 처리


## Requirements
* python 3.7.9
* pandas 1.1.3
* pytorch 1.5.0
* librosa 0.8.0
* unidecode 1.1.1
* inflect 0.2.5
* g2pk 0.9.4
* tensorboardX 2.0
* torchvision 0.6.0
* tensorboard 2.1.1
* matplotlib 3.3.2

## Preprocessing
To preprocess mel spectrogram and linear spectrogram magnitude, run this command:
<pre>
<code>
python prepare_data.py
</code>
</pre>

## Training
To train the model, run this command:
<pre>
<code>
python train_transformer.py
</code>
</pre>


## Evaluation
To evaluate, run:
<pre>
<code>
python synthesize.py --rhythm_scale=1.0 --restore_step1=500000
</code>
</pre>

## Results
Synthesized audio samples can be found in ./samples

## Reference code
* Transformer-TTS : https://github.com/soobinseo/Transformer-TTS
* SMART-Vocoder : https://github.com/SMART-TTS/SMART-G2P
* SMART-G2P : https://github.com/SMART-TTS/SMART-Vocoder
