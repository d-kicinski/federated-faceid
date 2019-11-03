class EarlyStopping:

    def __init__(self, patience: int):
        self._should_init = True
        self._patience: int = patience
        self._counter: int = 0
        self._should_break: bool = False
        self._value: float = 0.0

    @property
    def should_break(self) -> bool:
        return True if self._counter >= self._patience else False

    def is_best(self, value: float) -> bool:
        if self._should_init:
            return True

        if value < self._value:
            return True
        else:
            return False

    def update(self, loss: float) -> "EarlyStopping":
        if self._should_init:
            self._value = loss
            self._should_init = False

        if loss > self._value:
            self._counter += 1

        else:
            self._counter = 0
            self._value = loss

        return self
