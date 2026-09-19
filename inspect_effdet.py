import tflite

def inspect(model_path):
    with open(model_path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    print("Outputs from Subgraph:")
    for i in range(subgraph.OutputsLength()):
        tensor_idx = subgraph.Outputs(i)
        tensor = subgraph.Tensors(tensor_idx)

        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        print(f"  Out {i} (Tensor {tensor_idx} - {tensor.Name().decode()}): {shape}")

if __name__ == '__main__':
    inspect('assets/models/efficientdet_lite0.tflite')