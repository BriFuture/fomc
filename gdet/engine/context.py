import contextvars

class ContextManager:
    def __init__(self):
        self._epoch_var = contextvars.ContextVar("epoch", default=0)
        self._iter_var = contextvars.ContextVar("iter", default=0)

    def set_epoch(self, value):
        self._epoch_var.set(value)

    def get_epoch(self):
        return self._epoch_var.get()

    def set_iter(self, value):
        self._iter_var.set(value)

    def get_iter(self):
        return self._iter_var.get()