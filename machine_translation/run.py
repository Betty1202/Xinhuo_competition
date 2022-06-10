import os
from time import time

from onnxruntime import InferenceSession
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from machine_translation.generate_T5 import GenerativeT5

pretrained_model = 't5-base'
prompt = 'translate English to Chinese: I was a victim of a series of accidents.'


#
# # torch
# simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_model)
# tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
# generative_t5 = GenerativeT5(simplified_encoder, decoder_with_lm_head, tokenizer)
# print(generative_t5('translate English to Chinese: I was a victim of a series of accidents.', 16, temperature=0.)[0])
#
# # onnx
# decoder_sess = InferenceSession('/workspace/hxy/onnx/t5-decoder-with-lm-head-12.onnx')
# encoder_sess = InferenceSession('/workspace/hxy/onnx/t5-encoder-12.onnx')
# tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
# generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
# print(generative_t5('translate English to Chinese: I was a victim of a series of accidents.', 16, temperature=0.)[0])

# # torch
# model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#
# input_ids = tokenizer(prompt,
#                       return_tensors="pt").input_ids
# outputs = model.generate(input_ids, do_sample=False, max_length=30)
# print("0", tokenizer.batch_decode(outputs, skip_special_tokens=True))
#
# generative_t5 = GenerativeT5(model, tokenizer)
# print("1",
#       generative_t5(prompt, 16, temperature=0.)[0])
#
# # onnx
# model_session = InferenceSession('/workspace/hxy/onnx/t5-base.onnx')
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
# generative_t5 = GenerativeT5(model_session, tokenizer, onnx=True)
# print("2",
#       generative_t5(prompt, 16, temperature=0.)[0])
#
# # DTU
# model_sess = creat_DTU_session()
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
# generative_t5 = GenerativeT5_DTU(model_sess, tokenizer)
# print("3", generative_t5(prompt, 16, temperature=0.)[0])


class Inference():
    def __init__(self, type: str, pretrained_model):
        os.makedirs("onnx", exist_ok=True)
        if type == "torch":
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.generative_t5 = GenerativeT5(model, tokenizer)
        elif type == "onnx":
            if not os.path.exists(os.path.join("onnx", f"{pretrained_model}.onnx")):
                output = os.popen(f"python -m transformers.onnx --feature seq2seq-lm --model={pretrained_model} onnx/")
                output = os.popen(f"mv onnx/model.onnx onnx/{pretrained_model}.onnx")
            model_session = InferenceSession(os.path.join("onnx", f"{pretrained_model}.onnx"))
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.generative_t5 = GenerativeT5(model_session, tokenizer, onnx=True)
        elif type == "DTU":
            from machine_translation.engine_generator import creat_DTU_session
            from machine_translation.generate_T5.models import GenerativeT5_DTU
            if not os.path.exists(os.path.join("onnx", f"{pretrained_model}.onnx")):
                output = os.popen(f"python -m transformers.onnx --feature seq2seq-lm --model={pretrained_model} onnx/")
                output = os.popen(f"mv onnx/model.onnx onnx/{pretrained_model}.onnx")
            model_sess = creat_DTU_session(os.path.join("onnx", f"{pretrained_model}.onnx"))
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.generative_t5 = GenerativeT5_DTU(model_sess, tokenizer)
        else:
            raise NotImplementedError(f"Not yet implement {type}")

    def inference(self, prompt: str):
        start_time = time()
        result = self.generative_t5(prompt, 32, temperature=0.)[0]
        return result, time() - start_time


if __name__ == '__main__':
    temp = 'translate English to Chinese: I was a victim of a series of accidents.'
    model = "t5-base"
    inference = Inference("torch", "t5-base")
    print("torch", model, inference.inference(temp))
    inference = Inference("onnx", "t5-base")
    print("onnx", model, inference.inference(temp))
    inference = Inference("DTU", "t5-base")
    print("DTU", model, inference.inference(temp))
    inference = Inference("DTU", "Helsinki-NLP/opus-mt-en-zh")
    print("onnx", "Helsinki-NLP/opus-mt-en-zh", inference.inference(temp))
