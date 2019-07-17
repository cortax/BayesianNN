from dataprocessing import dataset_df_to_tensor, load_dfs
from torch.nn.functional import l1_loss, mse_loss

from torch import cat
import matplotlib.pyplot as plt

dfs = load_dfs('../DataBombardier')
taus = range(1,3500,50)

l1_losses = []

for tau in taus:

    datasets = [dataset_df_to_tensor(df, K=1, tau=tau, inputs_idx=(), with_y=True) for df in dfs]

    X = cat([dataset[0] for dataset in datasets])
    y = cat([dataset[1] for dataset in datasets])
    l1_losses.append(l1_loss(X,y))

    print(tau)

plt.plot(taus,l1_losses)
plt.xlabel('$\\tau$')
plt.ylabel('Mean of all $|y_t - y_{t+\\tau}|$')
plt.title('L1 losses for all $\\tau$')

plt.show()