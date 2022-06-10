# Xinhuo_competition

This repo generates multi-language caption for images. We have 2 versions of demo:

+ GUI version: Demo in this version with GUI, and you can run it on GPU or CPU with pt or onnx model. Note that, it
  can't run on DTU, because existing DTU environment doesn't support GUI.
+ DTU version: Demo in this version run on DTU.

## DTU version

### Environment

Please make sure that you have install following package in your environment:

```shell
pip install --upgrade pytorch torchvision
pip install onnxruntime

pip install transformers
pip install datasets
pip install sacrebleu
pip install sentencepiece

pip install scikit-image
```

### Inference

You can run this demo with the following script, and get the result in the last row of log:

```shell
python -m inference_DTU_total --language zh --type DTU
```

Note, you can change parameters as followings:

+ language: We support languages in [`en`, `zh`, `de`, `fr`, `ro`].
+ type: You can change type to `torch` or `onnx`, which infer the type of model without DTU.

## GUI version

TODO: Zewei