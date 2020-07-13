import tempfile
import shutil
import os
import torch
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def optimize(model, input, input_names, output_names, dynamic_axes):
    temp_file = os.path.join(tempfile.mkdtemp(), 'tmpmodel.onnx')
    #print(temp_file)
    onnx_opset = 10
    
    torch.onnx.export(model,                 # PyTorch model
                  input,                     # model input (or a tuple for multiple inputs)
                  temp_file,                 # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=onnx_opset,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=input_names,   # the model's input names
                  output_names=output_names, # the model's output names
                  dynamic_axes=dynamic_axes) # dynamic length axes

    optimized_model = OptimizedModel(temp_file)
    
    return optimized_model

class OptimizedModel(onnxruntime.InferenceSession):
    def __init__(self, filename):
        super().__init__(filename)
        self.temp_model_file = filename

    def forward(self, input):
        return self.run(None, {self.get_inputs()[0].name: to_numpy(input)})

    def serialize(self, model_file):
        shutil.copyfile(self.temp_model_file, model_file)
