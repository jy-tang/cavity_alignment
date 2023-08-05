function [x_new, xp_new] = lens(f, x, xp, displacement, angle)
 M = [1, 0;
      -1/f, 1];
 x0 = [x; xp];
 x1 = M*x0 + [0; displacement/f];
 x_new = x1(1);
 xp_new = x1(2);
 
end