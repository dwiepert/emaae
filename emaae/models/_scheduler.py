"""
Penalty scheduler classes

Author(s): Daniela Wiepert
Last modified: 02/10/2025
"""
# IMPORTS
## built-in
from typing import Optional 

class StepAlpha():
    """
    Scheduler that updates an alpha value (0-1) by gamma each call
    
    :param alpha: float, alpha value
    :param gamma: float, value to update alpha by each step
    """
    def __init__(self, alpha:float=0.25, gamma:float=0.1, epochs:Optional[int]=None):
        self.alpha = alpha
        if epochs is not None:
            self.gamma = 1 / (epochs-1) #FIGURE OUT GAMMA BASED ON EPOCHS
        else: 
            self.gamma = gamma
        self.step_counter = 0
    
    def step(self) -> float:
        """
        Update
        :param alpha: float, alpha value (0-1)
        :return new_alpha: float, updated alpha value(0-1)
        """
        new_alpha = self.alpha + self.gamma
        self.step_counter += 1
        if new_alpha > 1:
            new_alpha = 1
            print(f'Alpha value hit maximum (1) at step {self.step_counter}') 
        self.alpha = new_alpha
    
    def get_alpha(self) -> float:
        """
        :return: float, current alpha
        """
        return self.alpha 
    
    def get_step(self) -> int:
        """
        :return: int, current step
        """
        return self.step_counter