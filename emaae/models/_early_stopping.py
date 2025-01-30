"""
Early stopping
source: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
"""

class EarlyStopping:
    """
    patience: Number of epochs to wait before stopping if no improvement.
delta: Minimum change in the monitored quantity to qualify as an improvement.
best_score, best_model_state: Track the best validation score and model state.
call method: Updates the early stopping logic.
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_model = None

    def __call__(self, val_loss, model):
        """
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_model = model
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_model = model
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
    
    def get_best_model(self):
        return self.best_model