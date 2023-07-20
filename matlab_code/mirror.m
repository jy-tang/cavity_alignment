function [x_new, xp_new] = mirror(x,xp, displacement, angle)
    x0 = [x; xp];
    M = [-1, 0; 0, -1];
    x1 = M*x0 + [displacement; 2*angle];
    x_new = x1(1);
    xp_new = x1(2);
end