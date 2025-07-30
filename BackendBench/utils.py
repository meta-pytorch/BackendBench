import ast
import inspect
import re
import textwrap


def uses_cuda_stream(func) -> bool:
    """
    Detects whether a Python function creates CUDA streams.

    Args:
        func: The Python function to analyze

    Returns:
        bool: True if CUDA streams are created, False otherwise
    """
    try:
        source = inspect.getsource(func)
    except (TypeError, OSError):
        # Handle builtin functions, OpOverload objects, and other callables
        # without source code. These cannot create CUDA streams.
        return False

    # Check for stream creation patterns
    patterns = [
        r"torch\.cuda\.Stream\(",  # torch.cuda.Stream() constructor
        r"cupy\.cuda\.Stream\(",  # cupy.cuda.Stream() constructor
        r"cuda\.Stream\(",  # Generic cuda.Stream() constructor
        r"pycuda.*Stream\(",  # PyCUDA stream creation
        r"\bStream\(",  # Stream() constructor calls
        r"make_stream\(",  # make_stream() factory function
        r"create_stream\(",  # create_stream() factory function
    ]

    if any(re.search(p, source, re.IGNORECASE) for p in patterns):
        return True

    class StreamCreationFinder(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, node):
            # Check for Stream() constructor calls
            if hasattr(node.func, "attr") and node.func.attr == "Stream":
                self.found = True
            elif hasattr(node.func, "id") and node.func.id == "Stream":
                self.found = True
            self.generic_visit(node)

    tree = ast.parse(textwrap.dedent(source))
    finder = StreamCreationFinder()
    finder.visit(tree)
    return finder.found
