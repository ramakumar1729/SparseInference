function [ prox_x ] = prox_l1(x, use_l1)
% Soft thresholding operator for l1 norm
    prox_x = x;
    for i = 1:length(x)
       if x(i) > use_l1
           prox_x(i) = x(i) - use_l1;
       elseif x(i) < -use_l1
           prox_x(i) = x(i) + use_l1;
       else
           prox_x(i) = 0;
       end
    end
end