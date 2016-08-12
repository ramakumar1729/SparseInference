function [obj_val] = calc_obj(a_hat, A_potential, A_bad, C, num_cascades, i)

    num_nodes = size(A_potential,1);
    t_hat = zeros(num_cascades(i), 1);
 
    obj_val = 0;

    % ADD SI1 and SI2 CONTRIBUTION
    for j=1:num_nodes
        if (A_potential(j,i) > 0)
            obj_val = obj_val - a_hat(j) * (A_potential(j,i) + A_bad(j,i));
        end
    end
    
    % ADD SI3 CONTRIBUTION
    c_act = 1;
    for c=1:size(C, 1)
        idx = find(C(c,:)~=-1); % used nodes
        [val, ord] = sort(C(c, idx));
        idx_ord = idx(ord);
        cidx = find(idx_ord==i);
        
        if (~isempty(cidx) && cidx > 1)
            t_hat(c_act) = sum(a_hat(idx_ord(1:cidx-1)));
            obj_val = obj_val + log(t_hat(c_act));
            c_act = c_act + 1;
        end
    end
end