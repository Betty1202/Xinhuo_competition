from .run import Inference
import pandas as pd
import TopsInference

token_length = []
onnx_time = []
DTU_time = []

pretrain_model = "Helsinki-NLP/opus-mt-en-zh"
# inference_onnx = Inference("onnx", pretrain_model)
with TopsInference.device(0, 0):
    inference_DTU = Inference("DTU", pretrain_model)

    with open("Sentence_in_Flickr8k.txt", 'r') as f:
        while f:
            line = f.readline()
            if line == "":
                break
            # _, time = inference_onnx.inference(line)
            # onnx_time.append(time)
            _, time = inference_DTU.inference(line)
            DTU_time.append(time)
            token_length.append(len(inference_DTU.generative_t5.tokenizer(line)['input_ids']))

pd.DataFrame({"token_length": token_length, "onnx_time": DTU_time}).to_csv('time.csv')
