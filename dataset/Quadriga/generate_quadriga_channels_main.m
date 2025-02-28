% %% generate_quadriga_channels_main.m
% %
% % This main script generates channel matrices (H) using QuaDRiGa.
% % It creates one base station (8x8 UPA) and 20 user terminals (2x2 arrays)
% % for an outdoor 3GPP scenario. In addition to downtilt, it configures parameters
% % such as angular spreads, shadow fading, K-Factor, RMS delay spread, and spatial
% % consistency using helper functions. Then, it runs 1000 Monte Carlo simulations and
% % saves the channel matrix as a 4-D array with dimensions:
% % [Monte Carlo iteration, Tx antennas, User index, Rx antennas].
% %
% % Author: [Muhammad Ahmed Mohsin]
% % Date: [Feb 27, 2024]
% 
% %% Parameters and Setup
% clear; close all; clc;
% 
% % Create configuration object (ChannelConfig)
% config = ChannelConfig();
% disp('Current configuration:');
% disp(config);
% 
% %% Create Base Station Antenna (UPA 8x8)
% % Using the '3gpp-mmw' element type, build an 8x8 array.
% % The 5th and 6th parameters (here, 6 and config.downtiltAngle) denote the reference
% % downtilt and the actual downtilt, respectively.
% aBS = qd_arrayant('3gpp-mmw', 8, 8, [], 6, config.downtiltAngle, 0.5, 1, 2, 2.5, 2.5);
% aBS.combine_pattern;  % Pre-calculate the array pattern
% 
% %% Create Mobile Terminal (MT) Antenna (2x2 array)
% % Start with an omnidirectional element and replicate it to form a 2x2 array.
% aMT = qd_arrayant('omni');   % Create a single omni element
% aMT.copy_element(1,2);       % Duplicate to have 2 elements
% aMT.copy_element(1,3);       % Third element
% aMT.copy_element(1,4);       % Fourth element
% 
% % Set element positions to form a 2x2 grid (spacing = 0.5 wavelengths)
% % Positions must be a 3x4 matrix (each column = [x; y; z]).
% spacing = 0.5;
% aMT.element_position = [ 0,      spacing,  0,      spacing;
%                          0,      0,        spacing, spacing;
%                          0,      0,        0,       0 ];
% aMT.combine_pattern;
% 
% %% Set up the Layout and Scenario
% l = qd_layout;
% l.tx_position = [0; 0; config.BS_height];  % BS at origin, elevated by BS_height
% l.tx_array    = aBS;                        % Assign BS antenna
% 
% % Generate random user positions (2D Gaussian spread around BS)
% rng(1);  % For reproducibility
% rx_x = 100 * randn(1, config.numUsers);      % x-coordinates [m]
% rx_y = 100 * randn(1, config.numUsers);      % y-coordinates [m]
% rx_z = config.MT_height * ones(1, config.numUsers); % All users at MT_height
% l.rx_position = [rx_x; rx_y; rx_z];
% l.rx_array    = aMT;                        % Assign 2x2 MT antenna to each user
% l.no_rx = size(l.rx_position, 2);           % Explicitly set the number of receivers
% 
% % Choose an outdoor scenario – 3GPP UMa (urban macro)
% l.set_scenario('3GPP_38.901_UMa');
% 
% %% Visualize the Environment
% l.visualize;
% title('QuaDRiGa Layout Visualization');
% 
% %% Simulation Parameters
% s = qd_simulation_parameters;
% s.center_frequency = config.carrierFreq;  % Set carrier frequency
% s.use_3GPP_baseline = 1;                  % Use 3GPP baseline (disable spherical waves)
% s.show_progress_bars = 0;                 % Disable progress display
% l.simpar = s;
% 
% %% Initialize Builder and Configure Additional Parameters
% % Initialize the QuaDRiGa builder.
% b = l.init_builder;
% 
% % Configure additional parameters using helper functions.
% b = configure_angles(b, config.perClusterAS_A, config.perClusterAS_D, config.perClusterES_A, config.perClusterES_D);
% b = configure_sf_kf_ds(b, config.KF_mu, config.KF_sigma, config.DS_us, config.DS_sigma, config.SF_sigma);
% b = configure_spatial_correlation(b, config.SC_lambda);
% 
% %% Monte Carlo Simulation Loop
% numMC = 1000;
% % Determine dimensions:
% numTx = size(aBS.element_position, 2);      % Number of Tx antennas
% numRxAnt = size(aMT.element_position, 2);     % Number of Rx antennas
% 
% % Preallocate H_all array.
% % Desired dimensions: [Monte Carlo, Tx antennas, User index, Rx antennas]
% H_all = zeros(numMC, numTx, config.numUsers, numRxAnt);
% 
% for mc = 1:numMC
%     % Update the small-scale fading parameters.
%     gen_parameters(b);
% 
%     % Generate channel coefficients.
%     c = l.get_channels; 
%     % For multi-user layouts, get_channels typically returns a scalar structure.
%     % Its coeff field is expected to have dimensions:
%     % [numRxAnt, numTx, no_rx, no_freq]. We assume no frequency splitting (no_freq=1).
% 
%     % Debug: Display the size of c (only for first iteration)
%     if mc == 1
%         disp('Size of c:');
%         disp(size(c));
%         % Also display the size of the coeff field:
%         disp('Size of c(1).coeff:');
%         disp(size(c(1).coeff));
%     end
% 
%     % Extract the full coefficient array from c(1).coeff and squeeze singleton dimensions.
%     coeff_all = squeeze(c(1).coeff);
%     % Now, coeff_all should be of size: [numRxAnt, numTx, no_rx]
%     % We assume no_freq = 1, and no_rx should equal config.numUsers.
% 
%     for j = 1:config.numUsers
%         % Extract the channel matrix for user j.
%         % coeff_all(:,:,j) has dimensions [numRxAnt, numTx]
%         % Permute to obtain [numTx, numRxAnt].
%         H = permute(coeff_all(:,:,j), [2, 1]);
%         H_all(mc, :, j, :) = H;
%     end
% 
%     % Optionally, display progress every 50 iterations.
%     if mod(mc, 50) == 0
%         disp(['Monte Carlo iteration ' num2str(mc) ' out of ' num2str(numMC)]);
%     end
% end
% 
% %% Save the Monte Carlo Channel Data
% outputDir = 'Outputs';
% if ~exist(outputDir, 'dir')
%     mkdir(outputDir);
% end
% 
% save(fullfile(outputDir, 'Scenario_1.mat'), 'H_all', '-v7.3');
% disp(['Monte Carlo channel data (' num2str(numMC) ' iterations) saved in ' fullfile(outputDir, 'Scenario_1.mat')]);


%% generate_quadriga_channels_main.m
%
% This main script generates channel matrices (H) using QuaDRiGa.
% It creates one base station (8x8 UPA) and 20 user terminals (2x2 arrays)
% for an outdoor 3GPP scenario. In addition to downtilt, it configures parameters
% such as angular spreads, shadow fading, K-Factor, RMS delay spread, and spatial
% consistency using helper functions. Then, it runs 1000 Monte Carlo simulations and
% saves the following:
%   - H_all: a 4-D array with dimensions [MC iteration, Tx antennas, User index, Rx antennas]
%   - Delay_all: a cell array containing the delay information (from c.delay) for each simulation
%   - Par_all: a cell array containing the path parameters (from c.par) for each simulation
%
% These additional variables can later be used to compute and plot the distributions of
% azimuth/elevation angles, RMS delay spread, K-Factor, and shadow fading.
%
% Author: [Muhammad Ahmed Mohsin]
% Date: [Feb 27, 2024]

%% Parameters and Setup
clear; close all; clc;

% Create configuration object (ChannelConfig)
config = ChannelConfig();
disp('Current configuration:');
disp(config);

%% Create Base Station Antenna (UPA 8x8)
aBS = qd_arrayant('3gpp-mmw', 8, 8, [], 6, config.downtiltAngle, 0.5, 1, 2, 2.5, 2.5);
aBS.combine_pattern;

%% Create Mobile Terminal (MT) Antenna (2x2 array)
aMT = qd_arrayant('omni');
aMT.copy_element(1,2);
aMT.copy_element(1,3);
aMT.copy_element(1,4);
spacing = 0.5;
aMT.element_position = [ 0,      spacing,  0,      spacing;
                         0,      0,        spacing, spacing;
                         0,      0,        0,       0 ];
aMT.combine_pattern;

%% Set up the Layout and Scenario
l = qd_layout;
l.tx_position = [0; 0; config.BS_height];
l.tx_array    = aBS;

rng(1);  % For reproducibility
rx_x = 100 * randn(1, config.numUsers);
rx_y = 100 * randn(1, config.numUsers);
rx_z = config.MT_height * ones(1, config.numUsers);
l.rx_position = [rx_x; rx_y; rx_z];
l.rx_array    = aMT;
l.no_rx = size(l.rx_position, 2);  % Ensure the layout knows the number of receivers

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
Par_all   = cell(numMC, 1);

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
    Delay_all{mc} = c(1).delay;  % delay data (typically multi-dimensional)
    Par_all{mc}   = c(1).par;    % path parameter structure
    
    % Extract the full coefficient array and squeeze singleton dimensions.
    coeff_all = squeeze(c(1).coeff);  
    % Expected size: [numRxAnt, numTx, no_rx]
    % no_rx should be equal to l.no_rx (or config.numUsers)
    
    for j = 1:config.numUsers
        % Extract channel matrix for user j (dimensions: [numRxAnt, numTx])
        % Permute to obtain [numTx, numRxAnt]
        H = permute(coeff_all(:,:,j), [2, 1]);
        H_all(mc, :, j, :) = H;
    end
    
    if mod(mc, 50) == 0
        disp(['Monte Carlo iteration ' num2str(mc) ' out of ' num2str(numMC)]);
    end
end

%% Save the Monte Carlo Channel Data along with delay and parameter information
outputDir = 'Outputs';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'Scenario_1.mat'), 'H_all', 'Delay_all', 'Par_all', '-v7.3');
disp(['Monte Carlo channel data (', num2str(numMC), ' iterations) saved in ', fullfile(outputDir, 'Scenario_1.mat')]);
