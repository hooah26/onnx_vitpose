
import tensorrt as trt

import numpy as np

from PIL import Image
from torchvision.transforms import transforms
from models.model import ViTPose

from time import time
from configs.ViTPose_base_coco_256x192 import model as model_cfg

# Define file and model paths
IMG_PATH = "/workspace/ViTPose_pytorch/img1.jpg"
# ONNX_PATH = 'vitpose.onnx'
TRT_PATH = 'vitpose.trt'

# Load the model configuration
model = ViTPose(model_cfg)
device = next(model.parameters()).device
C, H, W = (3, 256, 192)

# Utility function to convert tensor to numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Convert ONNX model to TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#     with open(ONNX_PATH, 'rb') as model:
#         parser.parse(model.read())
#     engine = builder.build_cuda_engine(network)
#     with open(TRT_PATH, 'wb') as f:
#         f.write(engine.serialize())

# Create a runtime for inference
trt_runtime = trt.Runtime(TRT_LOGGER)
with open(TRT_PATH, 'rb') as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())

# Load and preprocess the input image
img = Image.open(IMG_PATH)
org_w, org_h = img.size
img_tensor = transforms.Compose(
    [transforms.Resize((H, W)),
     transforms.ToTensor()]
)(img).unsqueeze(0).to(device)

# Run inference on the image
tic = time()
context = engine.create_execution_context()
inputs = to_numpy(img_tensor).ravel()
outputs = np.empty([1, 17, 64, 48], dtype=np.float32)
bindings = [int(inputs.ctypes.data), int(outputs.ctypes.data)]
context.execute_v2(bindings)
elapsed_time = time() - tic
print(f">>> Output size: {outputs.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time ** -1: .1f} fps]\n")

# Continue the rest of your code
