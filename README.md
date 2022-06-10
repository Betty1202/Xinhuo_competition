# Xinhuo_competition

## Environment

```shell
pip install --upgrade pytorch torchvision
pip install onnxruntime

pip install transformers
pip install datasets
pip install sacrebleu
pip install sentencepiece
```

```shell
python -m transformers.onnx --feature seq2seq-lm --model=t5-base onnx/
```

## API

See example in `machine_translation.run.py`