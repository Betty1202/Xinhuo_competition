from onnxruntime import InferenceSession
from transformers import T5Tokenizer

from machine_translation.generate_T5 import create_t5_encoder_decoder, GenerativeT5

pretrained_model = 't5-base'

# torch
simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_model)
tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
generative_t5 = GenerativeT5(simplified_encoder, decoder_with_lm_head, tokenizer)
print(generative_t5('translate English to Chinese: Hello.', 16, temperature=0.)[0])

# onnx
decoder_sess = InferenceSession('/workspace/hxy/onnx/t5-decoder-with-lm-head-12.onnx')
encoder_sess = InferenceSession('/workspace/hxy/onnx/t5-encoder-12.onnx')
tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
print(generative_t5('translate English to Chinese: Hello.', 16, temperature=0.)[0])
