# vrnn
Variational Recurrent Autoencoder for Compustat Data

`pulldata.jl` pulls the Compustat data from WRDS via their SQL server to be used in `vrnn.jl`. 

`vrnn.jl` estimates an autoencoder for handling missing data in Compustat. 

Adapted from https://arxiv.org/abs/1506.02216