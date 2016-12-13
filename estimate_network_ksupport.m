function [A_hat, total_obj] = estimate_network_ksupport(A, C, num_nodes, horizon, type_diffusion, use_l2, use_l1, L, k)

num_cascades = zeros(1,num_nodes);
A_potential = sparse(zeros(size(A)));
A_bad = sparse(zeros(size(A)));
A_hat = sparse(zeros(size(A)));
total_obj = 0;

for c=1:size(C, 1),
    idx = find(C(c,:)~=-1); % used nodes
    [val, ord] = sort(C(c, idx));
    
    for i=2:length(val),
        num_cascades(idx(ord(i))) = num_cascades(idx(ord(i))) + 1;
        for j=1:i-1,
            if (strcmp(type_diffusion, 'exp'))
                A_potential(idx(ord(j)), idx(ord(i))) = A_potential(idx(ord(j)), idx(ord(i)))+val(i)-val(j);
            elseif (strcmp(type_diffusion, 'pl') && (val(i)-val(j)) > 1)
                A_potential(idx(ord(j)), idx(ord(i))) = A_potential(idx(ord(j)), idx(ord(i)))+log(val(i)-val(j));
            elseif (strcmp(type_diffusion, 'rayleigh'))
                A_potential(idx(ord(j)), idx(ord(i))) = A_potential(idx(ord(j)), idx(ord(i)))+0.5*(val(i)-val(j))^2;
            end
        end
    end
    
    for j=1:num_nodes,
        if isempty(find(idx==j))
            for i=1:length(val),
                if (strcmp(type_diffusion, 'exp'))
                    A_bad(idx(ord(i)), j) = A_bad(idx(ord(i)), j) + (horizon-val(i));
                elseif (strcmp(type_diffusion, 'pl') && (horizon-val(i)) > 1)
                    A_bad(idx(ord(i)), j) = A_bad(idx(ord(i)), j) + log(horizon-val(i));
                elseif (strcmp(type_diffusion, 'rayleigh'))
                    A_bad(idx(ord(i)), j) = A_bad(idx(ord(i)), j) + 0.5*(horizon-val(i))^2;
                end
            end
        end
    end
end

% we will have a convex program per column
for i=1:num_nodes,
    disp 'i:'
    i
    if (num_cascades(i)==0)
        A_hat(:,i) = 0;
        continue;
    end
    
    % 

    gamma = 0.1;
    iters_acc = 100;
    eps_acc = 1e-3;
	x0 = 0.0001*ones(num_nodes,1);
    
    [a_hat, obj] = overlap_nest(@calc_obj, @calc_grad, gamma, L, x0, k, iters_acc, eps_acc, A_potential, A_bad, C, num_cascades, i, use_l2, use_l1);
    % total_obj = total_obj + obj; % Sometimes, the procedure converges
    % within 99 iterations
    % assert(sum(a_hat>=0) == length(a_hat));
    % assert(sum(a_hat==inf) == 0);
    % assert(sum(isnan(a_hat)) ==  0);
    A_hat(:,i) = a_hat;
end
