import os
import sys

import TopsInference
import numpy as np
from machine_translation.utils import get_onnx_inputs_outputs

DEFAULT_INFO = {
    "encoder": {
        "onnx_path": "onnx/t5-encoder-12.onnx",
        "shapes": [[1, 256]]
    },
    "decoder": {
        "onnx_path": "onnx/t5-decoder-with-lm-head-12.onnx",
        "shapes": [[1, 256], [1, 256, 768]]
    },
    "base": {
        "onnx_path": "onnx/t5-base.onnx",
        "shapes": [[1, 256], [1, 256], [1, 256], [1, 256]]
    }
}


def generate_engine_file(model_path, inputs_info, output_names, cluster_id=0, precision='default'):
    model_name = model_path.split('/')[-1].replace('.onnx', '')

    PrecisionModeMap = {
        'default': TopsInference.KDEFAULT,
        'fp16': TopsInference.KFP16,
        'mix': TopsInference.KFP16_MIX,
    }

    if not os.path.exists("./engines"):
        os.mkdir("./engines")

    engine_model_path = './engines/%s_%s_bs%d.exec' % (
        model_name, precision, inputs_info['batchsize'])
    if os.path.isfile(engine_model_path) == True:
        print("Find engine file \'%s\'. Skip build engine." %
              (engine_model_path))
    else:
        input_names_str = ','.join(inputs_info['names'])
        input_shapes_str = ':'.join(
            [','.join([('%d' % dim) for dim in shape]) for shape in inputs_info['shapes']])
        input_types_str = ','.join(inputs_info['types'])
        output_names_str = ','.join(output_names)

        with TopsInference.device(0, cluster_id):
            onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
            if inputs_info['batchsize_modified']:
                print("[DEBUG] input names: %s" % (input_names_str))
                print("[DEBUG] input types: %s" % (input_types_str))
                print("[DEBUG] input shapes: %s" % (input_shapes_str))
                print("[DEBUG] output names: %s" % (output_names_str))

                onnx_parser.set_input_names(input_names_str)
                onnx_parser.set_input_dtypes(input_types_str)
                onnx_parser.set_input_shapes(input_shapes_str)
                onnx_parser.set_output_names(output_names_str)
                sys.stdout.flush()
            network = onnx_parser.read(model_path)
            # network.dump()
            optimizer = TopsInference.create_optimizer()
            optimizer.set_build_flag(PrecisionModeMap[precision])
            print("build engine ...")
            engine = optimizer.build(network)
            print("build engine finished.")
            engine.save_executable(engine_model_path)
            print("save engine file: %s" % (engine_model_path))
            engine = None
    return engine_model_path


def run(model_name, batchsize, precision):
    '''
    Run example
    :param model_name:
    :param batchsize:
    :param precision:
    :return:
    '''
    inputs_info, output_names = get_onnx_inputs_outputs(
        model_name, batchsize)

    #     inputs_info["shapes"] = [[1,256],[1,256,768]]
    inputs_info["shapes"] = [[1, 256]]

    engine_path = generate_engine_file(
        model_name, inputs_info, output_names, precision=precision)

    # generate random input data
    inputs_data = []
    for i in range(len(inputs_info['shapes'])):
        shape = inputs_info['shapes'][i]
        dtype = inputs_info['dtypes'][i]
        if dtype == np.float16:
            input_data = np.random.randn(*shape).astype(np.float16)
        elif dtype == np.float32:
            input_data = np.random.randn(*shape).astype(np.float32)
        elif dtype == np.float64:
            input_data = np.random.randn(*shape).astype(np.float64)
        elif dtype == np.bool:
            input_data = np.random.randint(2, size=shape, dtype=np.bool)
        elif dtype == np.int32:
            input_data = np.random.randint(2, size=shape, dtype=np.int32)
        elif dtype == np.int64:
            input_data = np.random.randint(2, size=shape, dtype=np.int64)
        else:
            assert 0, '[ERROR] unsupport dtype: {}'.format(dtype)
        inputs_data.append(input_data)

    with TopsInference.device(0, 0):
        engine = TopsInference.load(engine_path)
        outputs = []
        engine.run(inputs_data, outputs,
                   TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
    print("*" * 100)
    print("run successfully")


def create_t5_encoder_decoder_DTU():
    '''
    Create encoder and decoder for T5 on DTU
    :return:
    '''
    # Create encoder
    inputs_info, output_names = get_onnx_inputs_outputs(DEFAULT_INFO["encoder"]["onnx_path"], 1)
    inputs_info["shapes"] = DEFAULT_INFO["encoder"]["shapes"]
    encoder_engine_path = generate_engine_file(DEFAULT_INFO["encoder"]["onnx_path"], inputs_info, output_names,
                                               precision="default")

    # Create decoder
    inputs_info, output_names = get_onnx_inputs_outputs(DEFAULT_INFO["decoder"]["onnx_path"], 1)
    inputs_info["shapes"] = DEFAULT_INFO["decoder"]["shapes"]
    decoder_engine_path = generate_engine_file(DEFAULT_INFO["decoder"]["onnx_path"], inputs_info, output_names,
                                               precision="default")

    return encoder_engine_path, decoder_engine_path


def creat_DTU_session(onnx_path):
    '''
    Create engine file on DTU
    :return:
    '''
    inputs_info, output_names = get_onnx_inputs_outputs(onnx_path, 1)
    inputs_info["shapes"] = DEFAULT_INFO["base"]["shapes"]
    base_engine_path = generate_engine_file(onnx_path, inputs_info, output_names,
                                            precision="default")
    return base_engine_path


if __name__ == "__main__":
    #     model_path = "./onnx/t5-decoder-with-lm-head-12.onnx"
    model_path = "./onnx/t5-encoder-12.onnx"
    run(model_path, 1, "default")
