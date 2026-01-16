class Random_Policy:
    """Random policy for enviroment exploration"""
    def __init__(self, action_space, k:int=1):
        self.action_space = action_space
        self.old_a = None
        self.a_counter = 0
        self.k = k

    def act(self, x):
        
        if self.a_counter == self.k or self.old_a == None:
            self.a_counter = 0
            a = self.action_space.sample()
            self.old_a = a
            
        else:
            a = self.old_a
        
        self.a_counter += 1 
        
        return a
