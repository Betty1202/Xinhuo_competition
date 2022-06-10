import os
# os.environ['ENFLAME_LOG_LEVEL'] = 'ERROR'
# os.environ['ENFLAME_LOG_DEBUG_MOD'] = 'TOIR/ONNX'
# os.environ['SDK_LOG_LEVEL'] = '3'

import TopsInference
import numpy as np
import onnx
import sys
import time


PrecisionModeMap = {
    'default': TopsInference.KDEFAULT,
    'fp16': TopsInference.KFP16,
    'mix': TopsInference.KFP16_MIX,
}

ONNXDataTypeMap = {
    1: 'FLOAT',
    2: 'UINT8',
    3: 'INT8',
    4: 'UINT16',
    5: 'INT16',
    6: 'INT32',
    7: 'INT64',
    8: 'STRING',
    9: 'BOOL',
    10: 'FLOAT16',
    11: 'DOUBLE',
    12: 'UINT32',
    13: 'UINT64',
    14: 'COMPLEX64',
    15: 'COMPLEX128'
}


def get_onnx_inputs_outputs(model_name, batchsize=1):
    onnx_model = onnx.load(model_name)

    inputs = onnx_model.graph.input
    outputs = onnx_model.graph.output
    initializers = onnx_model.graph.initializer
    init_names = set([init.name for init in initializers])
    inputs_info = {}
    inputs_info['names'] = []
    inputs_info['shapes'] = []
    inputs_info['types'] = []
    inputs_info['dtypes'] = []
    inputs_info['batchsize_modified'] = False
    inputs_info['batchsize'] = batchsize
    output_names = []


    for _input in inputs:
        if _input.name in init_names:
            continue
#         print("[INFO] input type: ", _input.type)

        dims = _input.type.tensor_type.shape.dim
        shape = [dim.dim_value for dim in dims]

        if batchsize == -1:  # use default batchsize
            if type(shape[0]) != type(1):
                shape[0] = 1
                inputs_info['batchsize_modified'] = True
            elif shape[0] <= 0:
                shape[0] = 1
                inputs_info['batchsize_modified'] = True
            inputs_info['batchsize'] = shape[0]
        else:
            if shape[0] != batchsize:
                shape[0] = batchsize
                inputs_info['batchsize_modified'] = True

        inputs_info['names'].append(_input.name)
        onnx_data_type = ONNXDataTypeMap[_input.type.tensor_type.elem_type]
        if onnx_data_type == 'INT32':
            inputs_info['types'].append("DT_INT32")
            inputs_info['dtypes'].append(np.int32)
        elif onnx_data_type == 'INT64':
            inputs_info['types'].append("DT_INT64")
            inputs_info['dtypes'].append(np.int64)
        elif onnx_data_type == 'FLOAT16':
            inputs_info['types'].append("DT_FLOAT16")
            inputs_info['dtypes'].append(np.float16)
        elif onnx_data_type == 'BOOL':  # BOOL
            inputs_info['types'].append("DT_BOOL")
            inputs_info['dtypes'].append(np.bool)
        elif onnx_data_type == 'FLOAT':
            inputs_info['types'].append("DT_FLOAT32")
            inputs_info['dtypes'].append(np.float32)
        else:
            assert 0, '[ERROR] unsupport elem_type: {}'.format(onnx_data_type)
        print("input name: {}\tshape: {}\ttype: {}".format(
            _input.name, shape, onnx_data_type))

        inputs_info['shapes'].append(shape)

    for _output in outputs:
        print(f"output name: {_output.name}")
        output_names.append(_output.name)

    return inputs_info, output_names


def get_outputs_info(outputs_data):
    outputs_info = []
    for output_data in outputs_data:
        output_info = {'shape': output_data.shape, 'dtype': output_data.dtype,
                       'size': output_data.size * output_data.itemsize}
        outputs_info.append(output_info)
    return outputs_info
