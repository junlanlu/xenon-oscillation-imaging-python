close all;
load('../data/data_snrplot.mat');
data_static = data_static;
traj_static = traj_static;
npts = 128;
npts_evolve = npts;
npts_static = npts;
recon_size = 128;
subdiv_ind = 0;
i = 1;
j = 1;
nFrames = 1000;
snr_arr = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100, Inf];
key_radius_arr = 5;
for snr = snr_arr
    for key_radius = key_radius_arr
        
        load('../data/data_snrplot.mat');
        data = squeeze(data_evolve);
        data = data(nFrames*npts*subdiv_ind+1:nFrames*npts*(subdiv_ind + 1));
        traj = traj_evolve(nFrames*npts*subdiv_ind+1:nFrames*npts*(subdiv_ind + 1), :);
        % add noise to data
        if snr ~= inf
            data = awgn(data, snr, 'measured');
        end

        data_evolve = data;
        traj_evolve = traj;

        % Define important variables
        TR = 0.015; % in seconds

        traj_evolve = traj;
        traj_static = traj;

        % Prepare variables data and trajectory of keyhole analysis
        staticFID = conj(reshape(data_static, [npts_static, length(data_static) / npts_static]));
        evolveFID = conj(reshape(data_evolve, [npts_evolve, length(data_evolve) / npts_evolve]));
        traj = reshape(traj_evolve, [3, npts_evolve, length(data_evolve) / npts_evolve]);

        % Bin the wiggles
        [data_evolve_highkey, data_evolve_lowkey, data_evolve] ...
            = bin_wiggles(staticFID, evolveFID, traj, TR, key_radius, 'UFC', 'threshold_stretch');

        % remove data points without information
        high_zeros = find(data_evolve_highkey == 0);
        traj_evolve_highkey = traj_evolve;
        traj_evolve_highkey(high_zeros, :) = [];
        data_evolve_highkey(high_zeros) = [];
        [~, indices] = sort(abs(data_evolve_highkey));
        data_evolve_highkey = data_evolve_highkey(indices);
        traj_evolve_highkey = traj_evolve_highkey(indices, :);

        low_zeros = find(data_evolve_lowkey == 0);
        traj_evolve_lowkey = traj_evolve;
        traj_evolve_lowkey(low_zeros, :) = [];
        data_evolve_lowkey(low_zeros) = [];
        [~, indices] = sort(abs(data_evolve_lowkey));
        data_evolve_lowkey = data_evolve_lowkey(indices);
        traj_evolve_lowkey = traj_evolve_lowkey(indices, :);

        % Reconstruct
        reconVol_evolve_low = reconPhantomLowRes(recon_size, conj(data_evolve_lowkey(:)), traj_evolve_lowkey);
        reconVol_evolve_high = reconPhantomLowRes(recon_size, conj(data_evolve_highkey(:)), traj_evolve_highkey);
        reconVol_evolve_total = reconPhantomLowRes(recon_size, transpose(data_evolve(:))', traj_evolve);
        img_osc{i}{j} = ((abs(reconVol_evolve_high) - abs(reconVol_evolve_low))) ./ abs(reconVol_evolve_total);
        % increment index
        j = j + 1;
        close all;
    end
    j = 1;
    i = i + 1;
end
save('../data/analysis_snr.mat', "img_osc");

%% kspace to image space snr
mat_file = '../data/data_snrplot.mat';
i = 1;
for snr = snr_arr
    load(mat_file);
    data = squeeze(data);
    data = data(nFrames*npts*subdiv_ind+1:nFrames*npts*(subdiv_ind + 1));
    traj = traj(nFrames*npts*subdiv_ind+1:nFrames*npts*(subdiv_ind + 1), :);
    % add noise to data
    if snr ~= inf
        data = awgn(data, snr, 'measured');
    end
    reconVol_total = reconPhantomLowRes(recon_size, transpose(data(:))', traj);
    [imgSNR, imgSNR_Rayleigh, imgSignal, imgNoise] = imageSNR(abs(reconVol_total), mask, 8, 0.75*8^3);
    snr_img_rayleigh(i) = imgSNR_Rayleigh;
    snr_img(i) = imgSNR;
    i = i + 1;
end

figure(2);
plot(snr_arr(1:end-1), snr_img_rayleigh(1:end-1), 'k*--', 'LineWidth', 2);
hold on;
plot(snr_arr(1:end - 1), snr_arr(1:end - 1), 'k', 'LineWidth', 2)
pbaspect([1, 1, 1]);
xlim([0 100]);
ylim([0 100]);
grid on;
ax = gca;
ax.FontSize = 16;
hold off;
saveas_w(gcf, '../../tmp/snrcorrelation.png')

%% %%
load('../data/analysis_snr.mat');
load('../data/data_snrplot.mat')
%%
for i = 1:length(snr_arr) - 1
    img_osc_i = img_osc{i}{1};
    img_osc_inf = img_osc{end}{1};
    img_std1(i) = std(img_osc_i(mask_evolve == 0 & mask == 1));
    img_mean1(i) = mean(img_osc_i(mask_evolve == 0 & mask == 1));
    img_mean1_inf = mean(img_osc_inf(mask_evolve == 0 & mask == 1));
    img_meandiff1(i) = abs(img_mean1_inf-img_mean1(i));

    img_std2(i) = std(img_osc_i(mask_evolve > 0.09 & mask_evolve < 0.11));
    img_mean2(i) = mean(img_osc_i(mask_evolve > 0.09 & mask_evolve < 0.11));
    img_mean2_inf = mean(img_osc_inf(mask_evolve > 0.09 & mask_evolve < 0.11));
    img_meandiff2(i) = abs(img_mean2_inf-img_mean2(i));

    img_mean_tot(i) = mean(img_osc_i(mask > 0));
    img_mean_tot_inf = mean(img_osc_inf(mask > 0));
    img_meandiff_tot(i) = abs(img_mean_tot_inf-img_mean_tot(i));
end
figure(1);

plot(snr_img_rayleigh(find(snr_img_rayleigh < 40)), img_meandiff_tot(find(snr_img_rayleigh < 40)), 'k*--', 'LineWidth', 2);
hold on;
plot(snr_img_rayleigh(find(snr_img_rayleigh < 40)), img_meandiff1(find(snr_img_rayleigh < 40)), 'r*--', 'LineWidth', 2);
pbaspect([1, 1, 1]);
grid on;
ax = gca;
ax.FontSize = 16;
hold off;
saveas_w(gcf, 'figures/snrplot.png')
