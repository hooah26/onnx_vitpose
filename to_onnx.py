import os
import torch
import time

from models.model import ViTPose
from configs.ViTPose_base_coco_256x192 import model as model_cfg


CKPT_PATH = "/workspace/ViTPose_pytorch/vitpose-b-multi-coco.pth"

C, H, W = (3, 256, 192)

model = ViTPose(model_cfg)
ckpt = torch.load(CKPT_PATH)
model.load_state_dict(ckpt['state_dict'])
model.eval()

output_onnx = 'vitpose.onnx'
input_names = ["input_0"]
output_names = ["output_0"]

device = next(model.parameters()).device
inputs = torch.randn(1, C, H, W).to(device)

dynamic_axes = {'input_0' : {0 : 'batch_size'},
                'output_0' : {0 : 'batch_size'}}
start_time = time.time()

torch_out = torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=False,
                              input_names=input_names, output_names=output_names,
                              opset_version=11, dynamic_axes = dynamic_axes)
print(f">>> Saved at: {os.path.abspath(output_onnx)}")