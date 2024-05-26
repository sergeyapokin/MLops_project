import torch

from train import AirModel


def infer(X_test, y_test=None, checkpoint_path=None, trained_model=None):
    if checkpoint_path is not None:
        model = AirModel.load_from_checkpoint(checkpoint_path)
    elif trained_model is not None:
        model = trained_model
    else:
        raise NotImplementedError
    
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        y_pred = model(X_test)
        if y_test is not None:
            test_mse = loss(y_pred, y_test)
            print("test MSE %.4f" % (test_mse))
    return y_pred
