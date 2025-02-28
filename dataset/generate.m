%% Parameters and Setup
clear; close all; clc;

% Create configuration object (channel_config)
config = channel_config();
disp('Current configuration:');
disp(config);

%% Create Base Station Antenna (UPA 8x8)
aBS = qd_arrayant('3gpp-mmw', 8, 8, [], 6, config.downtiltAngle, 0.5, 1, 2, 2.5, 2.5);
aBS.combine_pattern;

%% Create Mobile Terminal (MT) Antenna (2x2 array)
aMT = qd_arrayant('omni');
aMT.copy_element(1, 2);
aMT.copy_element(1, 3);
aMT.copy_element(1, 4);
spacing = 0.5;
aMT.element_position = [0, spacing, 0, spacing;
                        0, 0, spacing, spacing;
                        0, 0, 0, 0];
aMT.combine_pattern;

%% Set up the Layout and Scenario
l = qd_layout;
l.tx_position = [0; 0; config.BS_height];
l.tx_array = aBS;

rng(1); % For reproducibility
rx_x = 100 * randn(1, config.numUsers);
rx_y = 100 * randn(1, config.numUsers);
rx_z = config.MT_height * ones(1, config.numUsers);
l.rx_position = [rx_x; rx_y; rx_z];
l.rx_array = aMT;
l.no_rx = size(l.rx_position, 2); % Ensure the layout knows the number of receivers

l.set_scenario('3GPP_38.901_UMa');

%% Visualize the Environment
l.visualize;
title('QuaDRiGa Layout Visualization');

%% Simulation Parameters
s = qd_simulation_parameters;
s.center_frequency = config.carrierFreq;
s.use_3GPP_baseline = 1;
s.show_progress_bars = 0;
l.simpar = s;

%% Initialize Builder and Configure Additional Parameters
b = l.init_builder;
b = configure_angles(b, config.perClusterAS_A, config.perClusterAS_D, config.perClusterES_A, config.perClusterES_D);
b = configure_sf_kf_ds(b, config.KF_mu, config.KF_sigma, config.DS_us, config.DS_sigma, config.SF_sigma);
b = configure_spatial_correlation(b, config.SC_lambda);

%% Monte Carlo Simulation Loop
numMC = 1000;
numTx = size(aBS.element_position, 2);
numRxAnt = size(aMT.element_position, 2);

% Preallocate composite channel array H_all:
% Desired dimensions: [MC iterations, Tx antennas, User index, Rx antennas]
H_all = zeros(numMC, numTx, config.numUsers, numRxAnt);

% Also prepare cell arrays for delays and path parameters:
Delay_all = cell(numMC, 1);
Par_all = cell(numMC, 1);

for mc = 1:numMC
    % Update small-scale fading parameters.
    gen_parameters(b);

    % Generate channel coefficients.
    c = l.get_channels;
    % Here, get_channels returns a scalar structure.
    % Its coeff field is assumed to have dimensions:
    % [numRxAnt, numTx, no_rx, no_freq]
    % In a multi-user setup, no_rx should equal the number of receiver positions.
    % For simplicity, we assume one frequency (or pick index 1).

    % Store delay and parameter information for the current Monte Carlo iteration.
    Delay_all{mc} = c(1).delay; % delay data (typically multi-dimensional)
    Par_all{mc} = c(1).par; % path parameter structure

    % Extract the full coefficient array and squeeze singleton dimensions.
    coeff_all = squeeze(c(1).coeff);
    % Expected size: [numRxAnt, numTx, no_rx]
    % no_rx should be equal to l.no_rx (or config.numUsers)

    for j = 1:config.numUsers
        % Extract channel matrix for user j (dimensions: [numRxAnt, numTx])
        % Permute to obtain [numTx, numRxAnt]
        H = permute(coeff_all(:, :, j), [2, 1]);
        H_all(mc, :, j, :) = H;
    end

    if mod(mc, 50) == 0
        disp(['Monte Carlo iteration ' num2str(mc) ' out of ' num2str(numMC)]);
    end

end

%% Save the Monte Carlo Channel Data along with delay and parameter information
outputDir = 'outputs';

if ~isfolder(outputDir)
    mkdir(outputDir);
end

save(fullfile(outputDir, 'scenario_1.mat'), 'H_all', 'Delay_all', 'Par_all', '-v7.3');
disp(['Channel data (', num2str(numMC), ' iterations) saved in ', fullfile(outputDir, 'scenario_1.mat')]);
