from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    """
    Data structure for single system transition 
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


if __name__ == "__main__":
    t = Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0]))
    print(t.state)