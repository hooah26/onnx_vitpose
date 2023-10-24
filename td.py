import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import torch
import time

#def to_numpy(tensor):
#    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

batch_size = 1
#x = torch.randn(batch_size, 224, 224, 3, requires_grad=True)
#input_data = to_numpy(x)
#input_data = np.zeros([1, 224, 224, 3], dtype=np.float32)
input_data = np.array(np.zeros([batch_size, 224, 224, 3]),dtype=np.float16)

infer_info = {
        #'url': 'archery-tritonserver:8095',
        'url': '0.0.0.0:8095',
        'verbose': False,
        'model_name': 'DEEPFACE',
        'model_version': '2',
        'input_name': 'zero_padding2d_input',
        'output_name': 'flatten',
        'input_shape':[1, 224, 224, 3],
        'dtype': 'FP16'
        }

start_time = time.time()
triton_client = httpclient.InferenceServerClient(
        url=infer_info['url'], verbose=infer_info['verbose'], concurrency=20)

inputs = [httpclient.InferInput(infer_info['input_name'], infer_info['input_shape'], infer_info['dtype'])]
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput(infer_info['output_name'])]



results = [triton_client.infer(infer_info['model_name'],
                                inputs=inputs,
                                outputs=outputs,
                                model_version=infer_info['model_version']
                                )]
end_time = time.time()
print('inference time: ', end_time - start_time)

#res = results[0].as_numpy(infer_info['output_name'])

