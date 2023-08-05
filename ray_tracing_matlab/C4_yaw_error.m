clear
close all

Lc = 65.2;        % cavity round-trip length
w = 0.6;          % width of the cavity
f = 16.3;         % focal length of the lenses




nroundtrip = 20;    % number of roundtrips     

angle_list = zeros(1, 6);   % yaw angle error on each of the optical elements
displacement_list = zeros(1,6); % displacement error on each of the optical elements

scan_delta_list = [0, 200e-9, 400e-9];
Legend = cell(length(scan_delta_list), 1);

count = 1;
for delta = scan_delta_list   % C4 yaw angle
 
    x = 0;            % initial (source) position in the center of the cavity
    xp = 0;             % initial (source) angle in the center of the cavity
    angle_list(6) = delta;
    
    x_source = zeros(1, nroundtrip + 1);  % record the center and angle at source and midpoint for each roundtrip
    xp_source = zeros(1, nroundtrip + 1);
    x_mid = zeros(1, nroundtrip);
    xp_mid = zeros(1, nroundtrip);

    x_source(1) = x;
    xp_source(1) = xp;
    
    for k = 1:nroundtrip

        [x, xp, x_m, xp_m] = cavity_roundtrip(x, xp, Lc, w, f, angle_list, displacement_list);
        x_source(k + 1) = x;
        xp_source(k + 1) = xp;
        x_mid(k) = x_m;
        xp_mid(k) = xp_m;


    end

    figure(1); 
    subplot(2,2,1);hold on
    plot(x_source*1e6, '.-')
    ylabel('Position $$\mu$$m','interpreter','latex')
    ylim([-15, 15])
    title('Source point')
    subplot(2,2,3);hold on
    plot(xp_source*1e6, '.-')
    ylim([-1, 1])
    ylabel('Angle $$\mu$$ rad','interpreter','latex')
    subplot(2,2,2);hold on
    plot(x_mid*1e6, '.-')
    ylim([-15, 15])
    ylabel('Position $$\mu$$m','interpreter','latex')
    title('Middle point')
    subplot(2,2,4);hold on
    plot(xp_mid*1e6, '.-')
    ylim([-1,1])
    ylabel('Angle $$\mu$$rad','interpreter','latex')
    
    Legend{count}=strcat('delta = ', num2str(delta));
    count = count + 1;
end

legend(Legend)
%%

%l = Lc;
%R11 = (8*f*(f-l) + l^2)/8/f^2;
%R12 = l*(32*f^2 - 12*f*l + l^2)/32/f^2;
%R21 = (l-4*f)/2/f^2;
%R22 = (8*f*(f-l) + l^2)/8/f^2;

%R = [R11, R12; R21, R22];

%R*[x0;xp0]

