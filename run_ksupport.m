%% RUN CODE TO GENERATE CONSTANTS

%TODO




%% CALL K-SUPPORT CODE
gamma = 0.1;
L = 100;
k = 4;
iters_acc = 100;
eps_acc = 1e-3;

[x, costs] = overlap_nest(@calc_obj, @calc_grad, gamma, L, x0, k, iters_acc, eps_acc);
