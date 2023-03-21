%% Create the phantom
% define the parameters
omega = 0.1; 
base_value = 1;
osc_amp1 = base_value * 0.05;
osc_amp2 = base_value * 0.05;
osc_amp3 = base_value * 0.05;
osc_amp4 = base_value * 0.1;
osc_amp5 = base_value * 0.1;
osc_amp6 = base_value * 0.1;
phase = 0 * pi/180; % in radians
imSz = [128, 128, 128];
reconSz = [128, 128, 128];
mat_file = '../data/demo_data.mat';
%% Calculate the trajectory
npts = 128;
dwell_time = 15;
ramp_time = 100;
plat_time = 2500;
decay_time = 60;
oversampling = 3;
del_x = 0;
del_y = 0;
del_z = 0;
flag = 3; % haltonized spiral
nFrames = 10000;
m_lNumberOfFrames = 1;

radialDistance_x = repelem(linspace(0, 1, npts), 3, 1)';
radialDistance_y = repelem(linspace(0, 1, npts), 3, 1)';
radialDistance_z = repelem(linspace(0, 1, npts), 3, 1)';

[x_end, y_end, z_end] = GX_f_gen_traj(nFrames, m_lNumberOfFrames, flag);
traj_scale_factor = 1;
x = traj_scale_factor * 0.5 * imSz(1) * radialDistance_x(:, 1) * x_end;
y = traj_scale_factor * 0.5 * imSz(2) * radialDistance_y(:, 2) * y_end;
z = traj_scale_factor * 0.5 * imSz(3) * radialDistance_z(:, 3) * z_end;

traj = reshape(z/imSz(1), [npts * nFrames, 1]);
traj(:, 2) = reshape(x/imSz(2), [npts * nFrames, 1]);
traj(:, 3) = reshape(y/imSz(3), [npts * nFrames, 1]);

%% Sample k-space
kspIm = zeros(npts, nFrames);
for i = 1:nFrames
    intensity1 = base_value + osc_amp1 * (square(i*omega + phase)-0.5);
    intensity2 = base_value + osc_amp2 * (square(i*omega + phase)-0.5);
    intensity3 = base_value + osc_amp3 * (square(i*omega + phase)-0.5);
    intensity4 = base_value + osc_amp4 * (square(i*omega + phase)-0.5);
    intensity5 = base_value + osc_amp5 * (square(i*omega + phase)-0.5);
    intensity6 = base_value + osc_amp6 * (square(i*omega + phase)-0.5);
    st = mri_objects('cyl3', [-.15, 0, -0.2, 0.1, 0.2, intensity1], ...
        'cyl3', [-0.15, 0, 0, 0.1, 0.2, intensity2], ...
        'cyl3', [-0.15, 0, 0.2, 0.1, 0.2, intensity3], ...
        'cyl3', [0.15, 0, .2, 0.1, 0.2, intensity4], ...
        'cyl3', [0.15, 0, 0, 0.1, 0.2, intensity5], ...
        'cyl3', [0.15, 0, -.2, 0.1, 0.2, intensity6]);
    kspIm_i = st.kspace(x(:, i), y(:, i), z(:, i));
    kspIm(:, i) = squeeze(kspIm_i);
end
data = reshape(kspIm, [npts * nFrames, 1]);

% define image space grid
[X, Y, Z] = meshgrid(linspace(-0.5, .5, imSz(1)), ...
    linspace(-0.5, .5, imSz(2)), ...
    linspace(-0.5, .5, imSz(3)));
image = st.image(X, Y, Z);
mask = imrotate3(double(image > osc_amp1 * 0.1), 90, [1, 0, 0]);
st = mri_objects('cyl3', [-.15, 0, -0.2, 0.1, 0.2, osc_amp1], ...
    'cyl3', [-0.15, 0, 0, 0.1, 0.2, osc_amp2], ...
    'cyl3', [-0.15, 0, 0.2, 0.1, 0.2, osc_amp3], ...
    'cyl3', [0.15, 0, .2, 0.1, 0.2, osc_amp4], ...
    'cyl3', [0.15, 0, 0, 0.1, 0.2, osc_amp5], ...
    'cyl3', [0.15, 0, -.2, 0.1, 0.2, osc_amp6]);
% get the image
image = st.image(X, Y, Z);
% get the image of the oscillation amplitude
mask_evolve = imrotate3(image, 90, [1, 0, 0]);
% save the data
save(mat_file, ...
    'traj', ...
    'data', ...
    'mask', ...
    'mask_evolve');
