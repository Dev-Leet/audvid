import tflite

def parse_tflite_signature(model_path):
    with open(model_path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)

    print(f"Num signatures: {model.SignatureDefsLength()}")
    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        print(f"Signature {i}:")

        print("  Inputs:")
        for j in range(sig.InputsLength()):
            tensor_map = sig.Inputs(j)
            print(f"    {tensor_map.Name().decode()}: Tensor {tensor_map.TensorIndex()}")

        print("  Outputs:")
        for j in range(sig.OutputsLength()):
            tensor_map = sig.Outputs(j)
            print(f"    {tensor_map.Name().decode()}: Tensor {tensor_map.TensorIndex()}")

if __name__ == '__main__':
    parse_tflite_signature('assets/models/movinet_a2_stream_int8.tflite')