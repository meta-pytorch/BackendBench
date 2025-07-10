# Sample Inputs Schema

This directory contains outputs of the huggingface tracer  which store traced PyTorch operation inputs from HuggingFace models.

'[hf_op_trace.json](https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/hf_op_trace.json)' contains an example of what these look like with the outputs from 20 models.

## Schema Structure

```json
{
  "operation_name": {
    "total_calls": <int>,
    "unique_input_count": <int>,
    "unique_inputs": [
      {
        "op_name": "<operation_name>",
        "input_shapes": [<shape_or_null>, ...],
        "input_dtypes": ["<dtype_string>", ...],
        "non_tensor_inputs": [<value_or_null_or_tensor_list_ref>, ...],
        "tensor_lists": {<tensor_list_metadata>},
        "count": <int>
      }
    ]
  }
}
```

## Field Descriptions

- **`input_shapes`**: List of tensor shapes (e.g., `[1, 3, 224, 224]`) or `null` for non-tensor inputs
- **`input_dtypes`**: List of type strings (e.g., `"torch.float32"`, `"<class 'int'>"`, `"<class 'list'>"`
- **`non_tensor_inputs`**: Actual non-tensor values, `null` for tensors, or `{"tensor_list_ref": <id>}` for tensor lists
- **`tensor_lists`**: Metadata for tensor lists, keyed by string IDs:
  ```json
  {
    "0": {
      "length": <int>,
      "shapes": [[<shape>], ...],
      "dtypes": ["<dtype>", ...]
    }
  }
  ```
- **`count`**: Frequency of this input combination in the traced data

**Note**: All dtypes (in input_dtypes and tensor_lists) are strings, not Python types (e.g., `torch.float32` instead of `float32`) as they are serialized in the JSON file. They should be converted to Python types before use.

## Examples

**Simple tensor input:**
```json
"input_shapes": [[2, 13]],
"input_dtypes": ["torch.int64"],
"non_tensor_inputs": [null]
```

**Tensor list input:**
```json
"input_shapes": [null, null],
"input_dtypes": ["<class 'list'>", "<class 'int'>"],
"non_tensor_inputs": [{"tensor_list_ref": 0}, 1],
"tensor_lists": {
  "0": {
    "length": 3,
    "shapes": [[1, 128, 20, 20], [1, 128, 20, 20], [1, 128, 20, 20]],
    "dtypes": ["torch.float32", "torch.float32", "torch.float32"]
  }
}
```

**Example entry with non-tensor inputs**
```json
"convolution.default": {
  "total_calls": 108,
  "unique_input_count": 67,
  "unique_inputs": [
    {
      "op_name": "convolution.default",
      "input_shapes": [[1, 256, 14, 14], [1024, 256, 1, 1], null, null, null, null, null, null, null],
      "input_dtypes": ["torch.float32", "torch.float32", "<class 'NoneType'>", "<class 'list'>", "<class 'list'>", "<class 'list'>", "<class 'bool'>", "<class 'list'>", "<class 'int'>"],
      "non_tensor_inputs": [null, null, null, [1, 1], [0, 0], [1, 1], false, [0, 0], 1],
      "tensor_lists": {},
      "count": 6
    },
    ...
  ]
  }
```
