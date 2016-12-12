function [x, curr_cost] = overlap_nest(f, gradf, gamma, L, x0, k, iters_acc, eps_acc, A_potential, A_bad, C, num_cascades, i, use_l2, use_l1)

% Minimizes regularization functional 
% f(w) + gamma/2 ||w||^2
% where ||.|| is the k overlap norm, or l2 &/or l1 norm
% uses Nesterov's accelerated method 
% L                : Lipschitz constant for gradf
% x0               : initial value
% iters_acc	   : maximum #iterations
% eps_acc	   : tolerance used as termination criterion  

t = 1;
alpha = x0;
x = x0;
d = length(x);
prev_cost = inf;
func_val = feval(f,x0,A_potential, A_bad, C, num_cascades,i, use_l2, use_l1);
if use_l1 == 0
    norm_val = ( norm_overlap(x0,k) )^2;
    curr_cost =  func_val + gamma/2 * norm_val;
else
    curr_cost = func_val;
end

theta = 1;

costs = zeros(iters_acc,1);

while (t < iters_acc && abs(prev_cost - curr_cost) > eps_acc)
  gradalpha = feval(gradf,alpha,A_potential, A_bad, C, num_cascades, i, use_l2);
  prevx = x;
  if use_l1 == 0
    x = prox_overlap( -1/L* gradalpha + alpha, k, L/gamma);
    theta = (sqrt(theta^4+4*theta^2)-theta^2)/2;
    rho = 1-theta+sqrt(1-theta);
    alpha = rho*x - (rho-1)*prevx;
  else
    x = prox_l1(-1/L* gradalpha + alpha, use_l1);
    alpha = x;
  end
  t = t+1;
  prev_cost = curr_cost;
  curr_cost = feval(f,alpha,A_potential, A_bad, C, num_cascades, i, use_l2, use_l1) + gamma/2 * ( norm_overlap(alpha,k) )^2;
  costs(t-1) = curr_cost;
end

