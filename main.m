network = 'kronecker-core-periphery-n1024-h10-r0_01-0_25-network.txt';
cascades = 'kronecker-core-periphery-n1024-h10-r0_01-0_25-1000-cascades.txt';

horizon = 10;
type_diffusion = 'exp';
num_nodes = 1024;
num_cascades = 10;
L = 1000;
k = 20;
use_l2 = 0;
use_l1 = 0;

[A_hat, A, total_obj, pr, mae, rmse] = netrate_ksupport(network, cascades, horizon, type_diffusion, num_nodes, use_l2, use_l1, L, k, num_cascades);

pr
mae
rmse
