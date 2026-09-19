import tflite

def generate_dart_mapping(model_path):
    with open(model_path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    # 1. Get the list of global tensor indices that make up the interpreter's inputs and outputs.
    # In tflite_flutter, getInputTensors()[i] corresponds to subgraph.Inputs(i)
    input_list = [subgraph.Inputs(i) for i in range(subgraph.InputsLength())]
    output_list = [subgraph.Outputs(i) for i in range(subgraph.OutputsLength())]

    # 2. Parse the signature to map logical names to global tensor indices
    sig = model.SignatureDefs(0)

    sig_inputs = {}
    for i in range(sig.InputsLength()):
        tensor_map = sig.Inputs(i)
        sig_inputs[tensor_map.Name().decode()] = tensor_map.TensorIndex()

    sig_outputs = {}
    for i in range(sig.OutputsLength()):
        tensor_map = sig.Outputs(i)
        sig_outputs[tensor_map.Name().decode()] = tensor_map.TensorIndex()

    # 3. Create the mapping from Dart input index to Dart output index
    # For every logical name in the signature (like 'state_block0_layer0_pool_buffer'),
    # find its Dart input index and Dart output index.

    print("final Map<int, int> stateInputToOutputIndex = {")
    for logical_name, global_in_idx in sig_inputs.items():
        if logical_name == 'image': continue # Skip the main input

        global_out_idx = sig_outputs.get(logical_name)
        if global_out_idx is None:
            print(f"    // WARNING: No output found for {logical_name}")
            continue

        # Find their positions in the Dart arrays
        dart_in_idx = input_list.index(global_in_idx)
        dart_out_idx = output_list.index(global_out_idx)

        print(f"  {dart_in_idx}: {dart_out_idx}, // {logical_name}")
    print("};")

    # Print the image/classifier indices too for confirmation
    image_global_idx = sig_inputs['image']
    logits_global_idx = sig_outputs['logits']
    print(f"// image input index: {input_list.index(image_global_idx)}")
    print(f"// classifier output index: {output_list.index(logits_global_idx)}")

if __name__ == '__main__':
    generate_dart_mapping('assets/models/movinet_a2_stream_int8.tflite')
