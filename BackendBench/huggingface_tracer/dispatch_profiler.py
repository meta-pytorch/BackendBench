import json
import logging
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)

MB = 1024 * 1024.0


class OpRecord:
    def __init__(
        self,
        op_name: str,
        input_shapes: list[tuple],
        output_shapes: list[tuple],
        time_taken_on_gpu: float,
        time_taken_on_cpu: float,
        non_tensor_inputs: list,
        memory_taken: float,
        input_dtypes: list[torch.dtype],
        tensor_lists: dict,
    ):
        self.op_name = op_name
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.time_taken_on_gpu = time_taken_on_gpu
        self.time_taken_on_cpu = time_taken_on_cpu
        self.memory_taken = memory_taken
        self.input_dtypes = [str(ele) for ele in input_dtypes]
        self.non_tensor_inputs = non_tensor_inputs
        self.tensor_lists = tensor_lists

    # for equivalence checking we only care about the op name, input shapes, input dtypes, and non tensor inputs
    def __hash__(self):
        # convert the lists and tuples into strings and hash them
        input_shapes_str = str(self.input_shapes)
        non_tensor_inputs_str = str(self.non_tensor_inputs)
        input_dtypes_str = str(self.input_dtypes)
        tensor_lists_str = str(self.tensor_lists)
        return hash(
            (
                self.op_name,
                input_shapes_str,
                input_dtypes_str,
                non_tensor_inputs_str,
                tensor_lists_str,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, OpRecord):
            return False

        # try:
        #     self.non_tensor_inputs == other.non_tensor_inputs
        # except:
        #     logger.info(
        #         f"the following is not checkable for equivalence: {self.non_tensor_inputs}"
        #     )
        #     logger.info(f"the ops are {self.summary()} \n and \n {other.summary()}")
        #     exit(1)

        return (
            self.op_name == other.op_name
            and self.input_shapes == other.input_shapes
            and self.input_dtypes == other.input_dtypes
            and self.non_tensor_inputs == other.non_tensor_inputs
            and self.tensor_lists == other.tensor_lists
        )

    def summary(self):

        # try:
        #     s = json.dumps(self.non_tensor_inputs)
        #     s = json.dumps(self.tensor_lists)
        # except:
        #     logger.info(
        #         f"the following is not json serializable: {self.non_tensor_inputs}"
        #     )
        #     logger.info(
        #         f"also possible that the following is not json serializable: {self.tensor_lists}"
        #     )
        #     exit(1)

        return {
            "op_name": self.op_name,
            "input_shapes": self.input_shapes,
            "input_dtypes": self.input_dtypes,
            "non_tensor_inputs": self.non_tensor_inputs,
            "tensor_lists": self.tensor_lists,
        }


class OpProfilerDispatchMode(TorchDispatchMode):

    # this is a dispatch mode that records the following:
    # 1. What aten op is being dispatched
    # 2. What is the input shape
    # 3. What is the output shape
    # 4. What is the time taken to dispatch the op
    # 5. What is the memory taken to dispatch the op

    def __init__(self):
        super().__init__()
        self.op_records = []

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        #  actually dispatch the op and get the result
        use_gpu = False
        start_time = time.time()
        rs = func(*args, **kwargs)
        end_time = time.time()
        mem: float = torch.cuda.memory_allocated() / MB
        #  record the op, input shape, output shape, time taken, memory taken
        input_shapes = []
        input_dtypes = []
        non_tensor_inputs = []
        tensor_lists = {}
        tensor_list_ind = 0

        if not torch.cuda.is_available():
            current_device = "cpu"
        else:
            current_device = torch.cuda.current_device()
        if isinstance(current_device, int) or "cuda" in current_device:
            cpu_start_time = time.time()
            torch.cuda.synchronize()
            cpu_end_time = time.time()
            time_taken_on_cpu = cpu_end_time - cpu_start_time
            use_gpu = True
        elif "cpu" in current_device:
            time_taken_on_gpu = 0
        else:
            raise ValueError(
                f"Unknown device: {current_device} right now we only support cpu and cuda"
            )

        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_shapes.append(arg.shape)
                input_dtypes.append(arg.dtype)
                non_tensor_inputs.append(None)
            elif isinstance(arg, list):
                # Handle lists
                input_shapes.append(None)
                input_dtypes.append(type(arg))
                if not any(isinstance(item, torch.Tensor) for item in arg):
                    # Empty list
                    non_tensor_inputs.append(arg)
                elif all(isinstance(item, torch.Tensor) for item in arg):
                    # All items are tensors - explode the list and put it in seperately to be reconstructed
                    non_tensor_inputs.append({"tensor_list_ref": tensor_list_ind})
                    tensor_dict = {
                        "length": len(arg),
                        "shapes": [item.shape for item in arg],
                        "dtypes": [str(item.dtype) for item in arg],
                    }
                    tensor_lists[tensor_list_ind] = tensor_dict
                    tensor_list_ind += 1

                    # Mixed types in list - create error
                else:
                    tensor_count = sum(
                        1 for item in arg if isinstance(item, torch.Tensor)
                    )
                    total_count = len(arg)
                    raise ValueError(
                        f"List contains mixed types: {tensor_count} tensors out of {total_count} items. "
                        f"Lists must contain either all tensors or no tensors. "
                        f"List contents: {[type(item).__name__ for item in arg]}"
                    )
            elif isinstance(arg, torch.dtype):
                input_shapes.append(None)
                input_dtypes.append(type(arg))
                non_tensor_inputs.append(str(arg))
            else:
                input_shapes.append(None)
                input_dtypes.append(type(arg))
                non_tensor_inputs.append(arg)

        output_shapes = []
        if isinstance(rs, torch.Tensor):
            output_shapes.append(rs.shape)
        elif isinstance(rs, (int, float)):
            output_shapes.append(())  # scalar shape
        else:
            output_shapes.append(None)

        if use_gpu:
            time_taken_on_gpu = end_time - start_time
        else:
            time_taken_on_cpu = end_time - start_time

        self.op_records.append(
            OpRecord(
                op_name=func.__name__,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                non_tensor_inputs=non_tensor_inputs,
                time_taken_on_gpu=time_taken_on_gpu,
                time_taken_on_cpu=time_taken_on_cpu,
                memory_taken=mem,
                input_dtypes=input_dtypes,
                tensor_lists=tensor_lists,
            )
        )
        return rs

    def get_op_records(self):
        return self.op_records


def main():
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
        torch.nn.Softmax(dim=1),
    )

    # Create sample input
    x = torch.randn(32, 10)

    # Enable profiling
    profiler = OpProfilerDispatchMode()
    with profiler:
        # Run model inference
        output = model(x)

    # Print profiling results
    print("\n=== Operation Profiling Results ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Get records from our custom profiler
    records = profiler.get_op_records()

    print("\nDetailed operation records:")
    for record in records:
        print(f"\nOperation: {record.op_name}")
        print(f"Input shapes: {record.input_shapes}")
        print(f"Output shapes: {record.output_shapes}")
        print(f"Time taken on gpu: {record.time_taken_on_gpu:.6f} seconds")
        print(f"Time taken on cpu: {record.time_taken_on_cpu:.6f} seconds")
        print(f"Memory used: {record.memory_taken:.2f} MB")


if __name__ == "__main__":
    main()
