import numpy as np

class Status():
    def __init__(self):
        self.best_train_loss =  np.Inf # track change in validation loss
        self.best_valid_loss =  np.Inf # track change in validation loss
        self.best_test_loss =  np.Inf # track change in validation loss
        self.best_f1_valid =  0
        self.best_f1_test =  0
        self.best_acc_test =  0
        self.best_bs =  0
        self.best_lr =  0
        self.n_epochs = 0
        self.patience = 0
        self.min_loss_improvement = 0
        self.epoch_model_saved = 0
        self.best_seed =  0
        self.optimizer = ''
        self.criterion = ''
        self.optimization_losses =  []
        self.best_train_losses = []
        self.best_valid_losses = []

    def print_summary(self):
        print("--------------------- SUMMARY ---------------------")
        print("LR of best result      : {}".format(self.best_lr))
        print("BS of best result      : {}".format(self.best_bs))
        print("Number of epochs       : {}".format(self.n_epochs))
        print("Used optimizer         : {}".format(self.optimizer))
        print("Used criterion         : {}".format(self.criterion))
        print("Seed of best result    : {}".format(self.best_seed))
        print("Lowest validation loss : {:.6f}".format(self.best_valid_loss))
        print("Best F1 valid          : {:.6f}".format(self.best_f1_valid))
        print("Test loss              : {:.6f}".format(self.best_test_loss))
        print("Best F1 test           : {:.6f}".format(self.best_f1_test))
        print("Best Accuracy test     : {:.6f}".format(self.best_acc_test))

    def clear(self):
        self.best_train_loss =  np.Inf # track change in validation loss
        self.best_valid_loss =  np.Inf # track change in validation loss
        self.best_test_loss =  np.Inf # track change in validation loss
        self.best_f1_valid =  0
        self.best_f1_test =  0
        self.best_acc_test =  0
        self.best_bs =  0
        self.best_lr =  0
        self.n_epochs = 0
        self.patience = 0
        self.min_loss_improvement = 0
        self.epoch_model_saved = 0
        self.best_seed =  0
        self.optimizer = ''
        self.criterion = ''
        self.optimization_losses =  []
        self.best_train_losses = []
        self.best_valid_losses = []