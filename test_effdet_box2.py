import tflite
import struct

def inspect(model_path):
    with open(model_path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    # EfficientDet-Lite0 anchor parsing check
    for i in range(subgraph.OutputsLength()):
        tensor_idx = subgraph.Outputs(i)
        tensor = subgraph.Tensors(tensor_idx)
        print(f"Output {i} is tensor {tensor.Name().decode()} with type {tensor.Type()}")

if __name__ == '__main__':
    inspect('assets/models/efficientdet_lite0.tflite')