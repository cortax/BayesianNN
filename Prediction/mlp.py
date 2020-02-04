import torch

def get_mlp(input_dim,layerwidth,nblayers,activation):
    param_count = (input_dim+1)*layerwidth+(nblayers-1)*(layerwidth**2+layerwidth)+layerwidth+1
    def mlp(x,theta,input_dim=input_dim,layerwidth=layerwidth,nb_layers=nblayers,activation=activation):
        """
        Feedforward neural network used as the observation model for the likelihood


        Parameters:
            x (Tensor): Input of the network of size NbExemples X NbDimensions   
            theta (Tensor):  M set of parameters of the network NbModels X NbParam
            input_dim (Int): dimensions of NN's inputs (=NbDimensions)
            layerwidth (Int): Number of hidden units per layer 
            nb_layers (Int): Number of layers
            activation (Module/Function): activation function of the neural network

        Returns:
            Predictions (Tensor) with dimensions NbModels X NbExemples X NbDimensions


        Example:

        input_dim=11
        nblayers = 2
        activation=nn.Tanh()
        layerwidth = 20
        param_count = (input_dim+1)*layerwidth+(nblayers-1)*(layerwidth**2+layerwidth)+layerwidth+1

        x=torch.rand(3,input_dim)
        theta=torch.rand(5,param_count)
        mlp(x,theta,input_dim=input_dim,layerwidth=layerwidth,nb_layers=nblayers,activation=activation)

        """

        nb_theta=theta.shape[0]
        nb_x=x.shape[0]
        split_sizes=[input_dim*layerwidth]+[layerwidth]+[layerwidth**2,layerwidth]*(nb_layers-1)+[layerwidth,1]
        theta=theta.split(split_sizes,dim=1)
        input_x=x.view(nb_x,input_dim,1)
        m=torch.matmul(theta[0].view(nb_theta,1,layerwidth,input_dim),input_x)
        m=m.add(theta[1].reshape(nb_theta,1,layerwidth,1))
        m=activation(m)
        for i in range(nb_layers-1):
            m=torch.matmul(theta[2*i+2].view(-1,1,layerwidth,layerwidth),m)
            m=m.add(theta[2*i+3].reshape(-1,1,layerwidth,1))
            m=activation(m)
        m=torch.matmul(theta[2*(nb_layers-1)+2].view(nb_theta,1,1,layerwidth),m)
        m=m.add(theta[2*(nb_layers-1)+3].reshape(nb_theta,1,1,1))
        return m.squeeze(-1)
    return param_count, mlp


