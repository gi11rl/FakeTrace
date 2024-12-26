# FakeTrace

## Env
```
conda env create -f requirements.yml
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Train
Config : `cfg/FakeTrace_train.yaml`

```
conda activate FakeTrace
python FakeTrace_train.py
```

## Test
### Enc
Config : `cfg/FakeTrace_enc.yaml`
```
conda activate FakeTrace
python FakeTrace_enc.py
```

### Dec
Config : `cfg/FakeTrace_dec.yaml`
```
conda activate FakeTrace
python FakeTrace_dec.py
```

## Ref
```
@inproceedings{wu2023sepmark,  
  title={SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection},  
  author={Wu, Xiaoshuai and Liao, Xin and Ou, Bo},  
  booktitle={Proceedings of the 31th ACM International Conference on Multimedia},  
  year={2023}  
}
```
