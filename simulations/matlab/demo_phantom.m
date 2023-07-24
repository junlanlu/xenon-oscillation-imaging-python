%% Create a phantom
omega = 0.1;
base_value = 1;
osc_amp1 = base_value * 0.1;
osc_amp2 = base_value * 0.1;
osc_amp3 = base_value * 0.1;
osc_amp4 = base_value * 0.1;
osc_amp5 = base_value * 0.1;
osc_amp6 = base_value * 0.1;
tr = 0.015; % in s
phase = 0 * pi/180; % in radians
reconstruct_flag = false;
mat_file = '../data/data_snr_uniform.mat';
waveform = "sine";
xcent = 0.17;
ycent = 0;
zcent = 0.2;
xrad = 0.13;
zwidth = 0.2;
%% Generate the trajectory
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
imSz = [npts, npts, npts];

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
    % square wave
    if waveform == "square"
        intensity1 = base_value + osc_amp1 * (square(i*omega + phase)-0.5);
        intensity2 = base_value + osc_amp2 * (square(i*omega + phase)-0.5);
        intensity3 = base_value + osc_amp3 * (square(i*omega + phase)-0.5);
        intensity4 = base_value + osc_amp4 * (square(i*omega + phase)-0.5);
        intensity5 = base_value + osc_amp5 * (square(i*omega + phase)-0.5);
        intensity6 = base_value + osc_amp6 * (square(i*omega + phase)-0.5);
    elseif waveform == "sine"
        intensity1 = base_value + osc_amp1 * (sin(i*omega + phase)-0.5);
        intensity2 = base_value + osc_amp2 * (sin(i*omega + phase)-0.5);
        intensity3 = base_value + osc_amp3 * (sin(i*omega + phase)-0.5);
        intensity4 = base_value + osc_amp4 * (sin(i*omega + phase)-0.5);
        intensity5 = base_value + osc_amp5 * (sin(i*omega + phase)-0.5);
        intensity6 = base_value + osc_amp6 * (sin(i*omega + phase)-0.5);
    else
        error('Invalid waveform');
    end
    st = mri_objects('cyl3', [-xcent, ycent, -zcent, xrad, zwidth, intensity1], ...
        'cyl3', [-xcent, ycent, 0, xrad, zwidth, intensity2], ...
        'cyl3', [-xcent, ycent, zcent, xrad, zwidth, intensity3], ...
        'cyl3', [xcent, ycent, -zcent, xrad, zwidth, intensity4], ...
        'cyl3', [xcent, ycent, 0, xrad, zwidth, intensity5], ...
        'cyl3', [xcent, ycent, zcent, xrad, zwidth, intensity6]);
    kspIm_i = st.kspace(x(:, i), y(:, i), z(:, i));
    kspIm(:, i) = squeeze(kspIm_i);
end
data = reshape(kspIm, [npts * nFrames, 1]);

kspIm = zeros(npts, nFrames);
for i = 1:nFrames
    intensity1 = base_value;
    intensity2 = base_value;
    intensity3 = base_value;
    intensity4 = base_value;
    intensity5 = base_value;
    intensity6 = base_value;
    st = mri_objects('cyl3', [-xcent, ycent, -zcent, xrad, zwidth, intensity1], ...
        'cyl3', [-xcent, ycent, 0, xrad, zwidth, intensity2], ...
        'cyl3', [-xcent, ycent, zcent, xrad, zwidth, intensity3], ...
        'cyl3', [xcent, ycent, -zcent, xrad, zwidth, intensity4], ...
        'cyl3', [xcent, ycent, 0, xrad, zwidth, intensity5], ...
        'cyl3', [xcent, ycent, zcent, xrad, zwidth, intensity6]);
    kspIm_i = st.kspace(x(:, i), y(:, i), z(:, i));
    kspIm(:, i) = squeeze(kspIm_i);
end
data_static = reshape(kspIm, [npts * nFrames, 1]);
%% Save mat file
[X, Y, Z] = meshgrid(linspace(-0.5, .5, imSz(1)), ...
    linspace(-0.5, .5, imSz(2)), ...
    linspace(-0.5, .5, imSz(3)));
image = st.image(X, Y, Z);
mask = imrotate3(double(image > 0), 90, [1, 0, 0]);
st = mri_objects('cyl3', [-xcent, ycent, -zcent, xrad, zwidth, osc_amp1], ...
    'cyl3', [-xcent, ycent, 0, xrad, zwidth, osc_amp2], ...
    'cyl3', [-xcent, ycent, zcent, xrad, zwidth, osc_amp3], ...
    'cyl3', [xcent, ycent, -zcent, xrad, zwidth, osc_amp4], ...
    'cyl3', [xcent, ycent, 0, xrad, zwidth, osc_amp5], ...
    'cyl3', [xcent, ycent, zcent, xrad, zwidth, osc_amp6]);
mask_evolve = st.image(X, Y, Z);
mask_evolve = double(imrotate3(mask_evolve, 90, [1, 0, 0]));
save(mat_file, ...
    'traj', ...
    'data', ...
    'data_static', ...
    'mask', ...
    'mask_evolve');
