import os
from time import time
import TopsInference
from onnxruntime import InferenceSession
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from machine_translation.generate_seq import GenerativeT5


class Inference():
    def __init__(self, type: str, pretrained_model):
        '''
        Inference machine translation
        :param type: model saved type. We support torch, onnx, DTU version.
        :param pretrained_model:
        '''
        os.makedirs("onnx", exist_ok=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        if type == "torch":
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.generative_t5 = GenerativeT5(model, tokenizer,
                                              decoder_start_token_id=model.config.decoder_start_token_id,
                                              eos_token_id=model.config.eos_token_id)
        elif type == "onnx":
            if not os.path.exists(os.path.join("onnx", f"{pretrained_model}.onnx")):
                output = os.popen(f"python -m transformers.onnx --feature seq2seq-lm --model={pretrained_model} onnx/")
                if "/" in pretrained_model:
                    output = os.popen(f"mkdir onnx/{'/'.join(pretrained_model.split('/')[:-1])}")
                output = os.popen(f"mv onnx/model.onnx onnx/{pretrained_model}.onnx")
            model_session = InferenceSession(os.path.join("onnx", f"{pretrained_model}.onnx"))
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.generative_t5 = GenerativeT5(model_session, tokenizer, onnx=True,
                                              decoder_start_token_id=model.config.decoder_start_token_id,
                                              eos_token_id=model.config.eos_token_id)
        elif type == "DTU":
            from DTU_inference.engine_generator import creat_DTU_LM_session
            from machine_translation.generate_seq.model_DTU import GenerativeT5_DTU
            if not os.path.exists(os.path.join("onnx", f"{pretrained_model}.onnx")):
                output = os.popen(f"python -m transformers.onnx --feature seq2seq-lm --model={pretrained_model} onnx/")
                output = os.popen(f"mv onnx/model.onnx onnx/{pretrained_model}.onnx")
            model_sess = creat_DTU_LM_session(os.path.join("onnx", f"{pretrained_model}.onnx"))
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.generative_t5 = GenerativeT5_DTU(model_sess, tokenizer,
                                                  decoder_start_token_id=model.config.decoder_start_token_id,
                                                  eos_token_id=model.config.eos_token_id)
        else:
            raise NotImplementedError(f"Not yet implement {type}")

    def inference(self, prompt: str):
        start_time = time()
        result = self.generative_t5(prompt, 32, temperature=0.)[0]
        return result, time() - start_time


if __name__ == '__main__':
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # input_ids = tokenizer("I'm Xinyi He",
    #                       return_tensors="pt").input_ids
    # outputs = model.generate(input_ids, do_sample=False, max_length=30)
    # print("0", tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # temp = 'translate English to Chinese: A group of people sit on a snowy mountain.'
    # pretrain_model = "t5-base"
    # # pretrain_model="Helsinki-NLP/opus-mt-en-zh"
    # inference = Inference("torch", pretrain_model)
    # print("torch", pretrain_model, inference.inference(temp))
    # inference = Inference("onnx", pretrain_model)
    # print("onnx", pretrain_model, inference.inference(temp))
    # inference = Inference("DTU", pretrain_model)
    # print("DTU", pretrain_model, inference.inference(temp))

    temp = 'A group of people sit on a snowy mountain'
    # pretrain_model = "t5-base"
    pretrain_model = "Helsinki-NLP/opus-mt-en-zh"
    # inference = Inference("torch", pretrain_model)
    # print("torch", pretrain_model, inference.inference(temp))
    # inference = Inference("onnx", pretrain_model)
    # print("onnx", pretrain_model, inference.inference(temp))
    with TopsInference.device(0, 0):
        inference = Inference("DTU", pretrain_model)
        print("DTU", pretrain_model, inference.inference(temp))
