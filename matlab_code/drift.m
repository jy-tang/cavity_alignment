function [x_new, xp_new] = drift(x,xp, L)
    M = [1, L; 0, 1];  
    x0 = [x;xp];
    x1 = M*x0;
    x_new = x1(1);
    xp_new = x1(2);
end