import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize(X=None, y=None, scaler_X=None, scaler_y=None):
    """[Apply transformation given by scalers to data]
    
    Keyword Arguments:
        X {[np.array]} -- [Input data] (default: {None})
        y {[np.array]} -- [Output data] (default: {None})
        scaler_X {[sklearn.preprocessing.Scaler]} -- [A fitted scaler for inputs] (default: {None})
        scaler_y {[sklearn.preprocessing.Scaler]} -- [A fitted scaler for output] (default: {None})
    
    Returns:
        [tuple] -- [transformed X and transformed y]
    """

    if scaler_X is not None and X is not None:
        tX = scaler_X.transform(X)
    else:
        tX = None

    if scaler_y is not None and y is not None:
        ty = scaler_y.transform(y)
    else:
        ty = None
    return tX, ty

    
def fitStandardScalerNormalization(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit_transform(X)
    scaler_y.fit_transform(y)

    return scaler_X, scaler_y