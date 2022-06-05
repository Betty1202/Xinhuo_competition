from onnxruntime import InferenceSession
from transformers import T5Tokenizer

from machine_translation.generate_T5 import create_t5_encoder_decoder, GenerativeT5
from machine_translation.engine_generator import create_t5_encoder_decoder_DTU

pretrained_model = 't5-base'

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

# DTU
encoder_sess, decoder_sess = create_t5_encoder_decoder_DTU()
tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, dtu=True)
print(generative_t5('translate English to Chinese: Hello.', 16, temperature=0.)[0])

# class Inference():
#     def __init__(self,type:str):
#         pretrained_model = 't5-base'
#         if type=="torch":
#             simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_model)
#             tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
#             self.generative_t5 = GenerativeT5(simplified_encoder, decoder_with_lm_head, tokenizer)
#         elif type=="onnx":
#             decoder_sess = InferenceSession('.onnx/t5-decoder-with-lm-head-12.onnx')
#             encoder_sess = InferenceSession('.onnx/t5-encoder-12.onnx')
#             tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
#             self.generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
#         elif type=="DTU":
