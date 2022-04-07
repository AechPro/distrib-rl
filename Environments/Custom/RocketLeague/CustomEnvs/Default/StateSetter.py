from rlgym.utils.state_setters import StateSetter, DefaultState, StateWrapper


class DefaultStateSetter(StateSetter):
    def __init__(self):
        super().__init__()
        self.setter = DefaultState()

    def reset(self, state_wrapper: StateWrapper):
        return self.setter.reset(state_wrapper)