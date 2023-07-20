function [x_source, xp_source, x_mid, xp_mid] = cavity_roundtrip(x, xp, Lc, w, f, angle_list, displacement_list)
    % angle_list, displacement: a list of 6 elements, for C1, L1, C2, C3, L2, C4
    % respectively
    
    %start from the source to C1
    L = (Lc- 2*w)/4;
    [x, xp] = drift(x,xp, L);
    
    %C1
    angle = angle_list(1);
    displacement = displacement_list(1);
    [x, xp] = mirror(x,xp, displacement, angle);
    
    %C1 to L1
    L = w/2;
    [x, xp] = drift(x, xp, L);
    
    %L1
    angle = angle_list(2);
    displacement = displacement_list(2);
    [x, xp] = lens(f, x, xp, displacement, angle);
    
    %L1 to C2
    L = w/2;
    [x, xp] = drift(x, xp, L);
    
    %C2
    angle = angle_list(3);
    displacement = displacement_list(3);
    [x, xp] = mirror(x,xp, displacement, angle);
    
    %C2 to mid
    L = (Lc- 2*w)/4;
    [x, xp] = drift(x, xp, L);
    x_mid = x; 
   	xp_mid = xp;
    
    %mid to C3
    L = (Lc- 2*w)/4;
    [x, xp] = drift(x, xp, L);
    
    %C3
    angle = angle_list(4);
    displacement = displacement_list(4);
    [x, xp] = mirror(x, xp, displacement, angle);
    
    % C3 to L2
    L = w/2;
    [x, xp] = drift(x, xp, L);
    
    % L2
    angle = angle_list(5);
    displacement = displacement_list(5);
    [x, xp] = lens(f, x, xp, displacement, angle);
    
    
    % L2 to C4
    L = w/2;
    [x, xp] = drift(x, xp, L);
    
    % C4
     angle = angle_list(6);
    displacement = displacement_list(6);
    [x, xp] = mirror(x, xp, displacement, angle);
    
    %C4 to source
    
    L = (Lc- 2*w)/4;
    [x, xp] = drift(x, xp, L);
    
    x_source = x; 
    xp_source = xp;
   
    
    
    
end