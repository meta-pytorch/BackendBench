from .base import Backend


class AtenBackend(Backend):
    def __init__(self) -> None:
        super().__init__("aten")

    def __getitem__(self, key):
        return key

    def __contains__(self, key):
        return True
