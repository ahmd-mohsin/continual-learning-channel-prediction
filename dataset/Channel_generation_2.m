%% Simplified UMa Simulation for 3GPP-5G
% This script runs a single-frequency Urban Macro (UMa) simulation using 3GPP-5G parameters.
% It generates a single BS layout, drops a reduced number of users, simulates the channel at 6 GHz,
% and plots the coupling loss, wideband SINR, and delay spread distributions.
%
% In addition, it extracts the full complex channel matrix H_full with dimensions:
%    (transmitter antennas, receiver antennas, number of users)
% For each user, we assume that cm(1,ir,1).coeff is an array that, after squeezing,
% is of size [N_rx x N_tx]. We then use permute to swap the dimensions to obtain a [N_tx x N_rx] matrix.
clear all; 
close all; 
warning('off','all');

%% Create Output Directory
if ~exist('output','dir')
    mkdir('output');
end
rng(0)
%% Antenna Setup
% Define port mapping for the BS antenna.
port_mapping = [1,0; 0,1; 1,0; 0,1; 1,0; 0,1; 1,0; 0,1];
port_mapping = [port_mapping, zeros(8,2); zeros(8,2), port_mapping] / 2;

% BS antenna configuration for UMa (12° downtilt)
aBS = qd_arrayant('3gpp-mmw', 4, 4, [], 6, 30, 0.5, 1, 2, 2.5, 2.5);
aBS.coupling = port_mapping;
aBS.combine_pattern;
aBS.element_position(1,:) = 0.8;  % Distance from the pole (meters)

% Mobile Terminal (MT) antenna configuration 
aMT = qd_arrayant('omni');
aMT.copy_element(1,2);
aMT.Fa(:,:,2) = 0;
aMT.Fb(:,:,2) = 1;

%% Simulation Parameters
no_rx = 700;                   % Number of mobile terminals
s = qd_simulation_parameters;  % General simulation parameters
s.center_frequency = 8e9;      % Single carrier frequency: 6 GHz
no_freq = 1;                   % Only one frequency is simulated
s.use_3GPP_baseline = 1;       % Use 3GPP baseline (disable spherical waves)
s.show_progress_bars = 1;      % Disable progress bars

%% Layout Generation: Single BS UMa
ISD = 700;  % Inter-site distance (sets the scale)
l = qd_layout.generate('regular', 1, ISD, aBS);
l.simpar = s;
l.tx_position(3,:) = 20;       
l.name = 'UMa';

%% Drop Users
l.no_rx = no_rx;
no_go_dist = 40;             
ind = true(1, no_rx);
while any(ind)
    l.randomize_rx_positions(0.93*ISD, 1.5, 1.5, 0, ind);
    ind = sqrt(l.rx_position(1,:).^2 + l.rx_position(2,:).^2) < no_go_dist;
end
% Set user heights (outdoor users: 1.5 m tall)
l.rx_position(3,:) = 2;
% Assign UMa scenario parameters (80% indoor probability; outdoor users set to 1.5 m)
indoor_rx = l.set_scenario('3GPP_38.901_UMa', [], [], 0.8);
l.rx_position(3, ~indoor_rx) = 2.0;
l.rx_array = aMT;

%% Channel Generation
% Initialize channel builder and disable spatial consistency.
b = l.init_builder;
for ib = 1:numel(b)
    [i1, i2] = qf.qind2sub(size(b), ib);
    scenpar = b(i1,i2).scenpar;
    scenpar.SC_lambda = 0.4;  % Disable spatial consistency for SSF
    b(i1,i2).scenpar_nocheck = scenpar;
end
b = split_multi_freq(b);  % Split builder per frequency (here only one)
gen_parameters(b);        % Generate LSF and SSF parameters
cm = get_channels(b);     % Generate channels

%% Determine Dimensions for Full Channel Extraction
% Use the full BS antenna array from l.tx_array.
N_tx = l.tx_array.no_elements;   % Expected to be 16 (from a 4x4 array)
N_rx = aMT.no_elements;           % Expected to be 2
% For a single-site layout, cm is assumed to be of size [1, no_rx, no_freq].

%% Extract Full Complex Channel Matrix H_full
% We want H_full to have size (N_tx, N_rx, no_rx).
% For each user (ir from 1 to no_rx), extract the coefficient from cm(1,ir,1).coeff.
% After squeezing, we expect this to be of size [N_rx x N_tx]. We then swap dimensions.
H_full = zeros(N_tx, N_rx, no_rx);  % Preallocate H_full

for ir = 1:no_rx
    % Access the channel for user ir. Here, cm is indexed as (1,ir,1) because the first dimension corresponds to the single BS.
    tmp = squeeze(cm(1,ir,1).coeff);
    % If tmp is still an N-D array with more than 2 dimensions, take the first page.
    if ndims(tmp) > 2
        tmp = tmp(:,:,1);
    end
    % Now, tmp should be of size [N_rx x N_tx]. We want a [N_tx x N_rx] matrix.
    H_full(:,:,ir) = permute(tmp, [2, 1]);
end

% Save the extracted full channel matrix.
save('output/H_full.mat','H_full');

%% Plotting Results

% (A) Coupling Loss CDF
figure('Position',[50, 550, 1400, 640]);
cl = zeros(1, no_rx);
bins_cl = -210:0.5:-60;  % Bins in dB
for ir = 1:no_rx
    H_temp = squeeze(cm(1,ir,1).coeff);
    if ndims(H_temp) > 2
        H_temp = H_temp(:,:,1);
    end
    % H_temp is expected to be [N_rx x N_tx]. Compute effective power per TX element.
    pg_eff = zeros(1, N_tx);
    for j = 1:N_tx
        pg_eff(j) = sum(abs(H_temp(:,j)).^2) / 8;
    end
    cl(ir) = 10*log10(max(pg_eff));
end
plot(bins_cl, 100*qf.acdf(cl, bins_cl), 'b-', 'LineWidth',2);
grid on; box on;
xlabel('Coupling Loss (dB)');
ylabel('CDF [%]');
title(['Coupling Loss CDF - ', l.name]);

% (B) Wide-band SINR CDF
figure('Position',[50, 550, 1400, 640]);
sinr = zeros(1, no_rx);
bins_sinr = -10:0.5:40;
for ir = 1:no_rx
    H_temp = squeeze(cm(1,ir,1).coeff);
    if ndims(H_temp) > 2
        H_temp = H_temp(:,:,1);
    end
    rsrp_p0 = zeros(1, N_tx);
    for j = 1:N_tx
        tmp = H_temp(:,j);  % Column vector for TX element j
        rsrp_p0(j) = sum(abs(tmp).^2) / 2;  % Divide by N_rx (2)
    end
    sinr(ir) = 10*log10(max(rsrp_p0) / (sum(rsrp_p0) - max(rsrp_p0)));
end
plot(bins_sinr, 100*qf.acdf(sinr, bins_sinr), 'r-', 'LineWidth',2);
grid on; box on;
xlabel('Wideband SINR (dB)');
xlim([-10 10])
ylabel('CDF [%]');
title(['Wideband SINR CDF - ', l.name]);

% (C) Delay Spread CDF
figure('Position',[50, 550, 1400, 640]);
ds = zeros(1, no_rx);
bins_ds = 0:5:500;   % Delay spread bins in nanoseconds
for ir = 1:no_rx
    H_temp = squeeze(cm(1,ir,1).coeff);
    if ndims(H_temp) > 2
        H_temp = H_temp(:,:,1);
    end
    pg_eff = zeros(1, N_tx);
    ds_tmp = zeros(1, N_tx);
    for j = 1:N_tx
        pg_eff(j) = sum(abs(H_temp(:,j)).^2) / 8;
        ds_tmp(j) = cm(1,ir,1).par.ds_cb;  % Assume ds_cb is the same for all TX elements
    end
    [~, idx] = max(pg_eff);
    ds(ir) = ds_tmp(idx) * 1e9;  % Convert seconds to nanoseconds
end
plot(bins_ds, 100*qf.acdf(ds, bins_ds), 'm-', 'LineWidth',2);
grid on; box on;
xlabel('Delay Spread (nsec)');
ylabel('CDF [%]');
title(['Delay Spread CDF - ', l.name]);

%% End of Script
disp('Simulation complete. The full channel matrix H_full (Tx, Rx, users) is saved in output/H_full.mat');
