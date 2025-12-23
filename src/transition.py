class Transition:
    def __init__(self, x_t, a_t, r_t, x_tn) -> None:
        self.state = x_t
        self.action = a_t 
        self.reward = r_t
        self.next_state = x_tn 
