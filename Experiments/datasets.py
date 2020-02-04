import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Ne pas oublier d'ajuster le code dépendamment de si les targets sont placés au début ou à la fin du fichier. Aussi, changer le délimiteur au besoin.

# Exemple de commande: make_datasets('redwine', 0.9, StandardScaler(), 5)

def load_data(txt, size_train):

#    data = np.loadtxt(txt + '/' + txt + '.txt')
    data = np.genfromtxt(txt + '/' + txt + '.csv', delimiter=';')
    
    # Les targets sont à la fin. 
    X = data[:, range(data.shape[1] - 1) ]
    y = data[:, data.shape[1] - 1]
    
    # Les targets sont au début. 
#    X = data[:, 0:]
#    y = data[:, 0]


    size_train = np.round(X.shape[0] * size_train)
    permutation = np.random.choice(range (X.shape[0]), X.shape[0], replace = False)
    index_train = permutation[0 : int(size_train)]
    index_test = permutation[int(size_train) : ]

    X_train = X[index_train, :]
    y_train = y[index_train]
    X_test = X[index_test, :]
    y_test = y[index_test]

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).unsqueeze(-1)
    
    return X_train, y_train, X_test, y_test
    
    
def make_datasets(txt, size_train, scaler, n_splits):    
    
    for i in range(0, n_splits):
        
        X_train, y_train, X_test, y_test = load_data(txt, size_train)
        
        torch.save(X_train, txt + '/data/' + txt + '_X_train_(' + str(i) + ').pt')
        torch.save(y_train, txt + '/data/' + txt + '_y_train_(' + str(i) + ').pt')
        torch.save(X_test, txt + '/data/' + txt + '_X_test_(' + str(i) + ').pt')
        torch.save(y_test, txt + '/data/' + txt + '_y_test_(' + str(i) + ').pt')
        
def normalize(X_train, y_train, X_test, y_test, scaler):

    scaler_X = scaler
    scaler_y = scaler
    inverse_scaler_y = scaler_y.inverse_transform

    X_train = torch.as_tensor(scaler_X.fit_transform(X_train)).float()
    y_train = torch.as_tensor(scaler_y.fit_transform(y_train)).float()

    X_test = torch.as_tensor(scaler_X.transform(X_test)).float()

    return X_train, y_train, X_test, y_test.float(), inverse_scaler_y