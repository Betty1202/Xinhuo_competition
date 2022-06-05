import os

import TopsInference

os.environ['ENFLAME_LOG_LEVEL'] = 'FATAL'
os.environ['SDK_LOG_LEVEL'] = '3'

# %% md

### 3.9 engineTool定义

'''
`engineTool`为一个类，onnx模型路径，板卡ID和该板卡推理是调用的cluster id，该类主要包含了以下方法：
1. `get_engine`：输入为数据精度和输入大小，返回值为engine所在路径，当检测到engine不存在时，会自动通过onnx模型来生成engine，并保存在`/tmp/engines/`路径下。

需要注意的是，由于onnx模型的batchsize是动态的，因此在生成engine的时候需要指定engine输入的形状，将动态输入转为静态输入。
'''


# %%

class engineTool:
    TopsInference.create_error_manager()

    def __init__(self, model_path, device_id, cluster_id):
        self.model_path = model_path
        self.device_id = device_id
        self.cluster_id = cluster_id

    def get_engine(self, precision, shape=None):

        if not os.path.exists('/tmp/engines/'):
            os.mkdir('/tmp/engines/')

        # generate engine name with path
        model_name = self.model_path.split('/')[-1].replace('.onnx', '')
        engine_model_name = '/tmp/engines/' + model_name + ('_{}.exec').format(precision)
        print('engine file: {}'.format(engine_model_name))

        # if the engine exists, return the engint path, or generate a new engine and return its path
        if os.path.isfile(engine_model_name) == True:
            return engine_model_name

        else:
            if os.path.isfile(self.model_path) != True:
                assert 0, "Fail to load model file: {}".format(self.model_path)
            print("running {}".format(self.model_path))

            if precision == 'default':
                precision_mode = TopsInference.KDEFAULT
            elif precision == 'fp16':
                precision_mode = TopsInference.KFP16
            elif precision == 'fp16_mix':
                precision_mode = TopsInference.KFP16_MIX
            else:
                assert False, "unknown precision mode: {}".format(precision)

            with TopsInference.device(self.device_id, self.cluster_id):
                onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
                if shape:
                    onnx_parser.set_input_shapes(shape)
                model = onnx_parser.read(self.model_path)
                optimizer = TopsInference.create_optimizer()
                # config the engine precision
                optimizer.set_build_flag(precision_mode)
                print("build engine ...")
                engine = optimizer.build(model)
                print("build engine finished.")
                engine.save_executable(engine_model_name)
                print("save engine file: {}".format(engine_model_name))
                return engine_model_name


from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")

DEVICE_ID = 0
CLUSTER_IDS = 0
model_path = "/workspace/hxy/onnx/model.onnx"

precision = "fp16"

et = engineTool(model_path, device_id=0, cluster_id=0)
engine_path = et.get_engine(precision=precision)

with TopsInference.device(0, 0):
    engine = TopsInference.load(engine_path)
    inf_result = []
    engine.run([dict(inputs)], inf_result, TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)

print(inf_result)
