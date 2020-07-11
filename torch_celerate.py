import onnxruntime

def optimize(ptmodel, input, input_names, output_names, dynamic_axes,

model,
                                                    x,
                                                    input_names = ['input'],
                                                    output_names = ['output'],
                                                    dynamic_axes={'input' : {0 : 'batch_size'},
                                                    'output' : {0 : 'batch_size'}})
                                                    
