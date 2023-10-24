import numpy as np
import tritonclient.http as httpclient
from torchvision.transforms import transforms
from PIL import Image

from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import torch
import time 

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

batch_size = 1
x = torch.randn(batch_size, 3 , 256, 192, requires_grad=True)
input_data = np.array(to_numpy(x), dtype=np.float16)


# IMG_PATH = "/workspace/ViTPose_pytorch/img1.jpg"
# C, H, W = (3, 256, 192)
#
# img = Image.open(IMG_PATH)
# org_w, org_h = img.size
# img_tensor = transforms.Compose([transforms.Resize((H, W)),transforms.ToTensor()])(img).unsqueeze(0).to(device)
# input_data = to_numpy(img_tensor).ravel()

infer_info = {
        #'url': 'archery-tritonserver:8000',
        'url': '0.0.0.0:8095',
        'verbose': False,
        'model_name': 'VITPOSE',
        'model_version': '1',
        'input_name': 'input_0',
        'output_name': 'output_0',
        'input_shape':[1, 3, 256, 192],
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

res = results[0].as_numpy(infer_info['output_name'])
