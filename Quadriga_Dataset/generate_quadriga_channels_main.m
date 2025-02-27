% % generate_quadriga_channels.m
% %
% % This script generates channel matrices (H) using QuaDRiGa.
% % It creates one base station (8x8 UPA) and 20 user terminals (2x2 arrays)
% % for an outdoor 3GPP scenario. The downtilt angle at the transmitter is
% % specified by the variable downtiltAngle.
% %
% 
% %% Parameters and Setup
% clear; close all; clc;
% 
% % ----- User-adjustable Parameter: Downtilt angle (in degrees) -----
% downtiltAngle = 120; % Change this value for different datasets
% 
% % ----- Simulation Parameters -----
% numUsers      = 20;       % Number of user terminals
% BS_height     = 25;       % Base station height (z-coordinate)
% MT_height     = 1.5;      % User terminal (MT) height
% carrierFreq   = 3.5e9;    % Carrier frequency [Hz]
% 
% % ----- Monte Carlo simulation: one realization (loop if needed) -----
% 
% 
% %% Create Base Station Antenna (UPA 8x8)
% % Using the '3gpp-mmw' element type, we build an 8x8 array.
% % Note: The 5th and 6th parameters in qd_arrayant (here, 6 and downtiltAngle)
% % denote the electrical downtilt reference and the downtilt, respectively.
% aBS = qd_arrayant('3gpp-mmw', 8, 8, [], 6, downtiltAngle, 0.5, 1, 2, 2.5, 2.5);
% aBS.combine_pattern;  % Pre-calculate the array pattern
% 
% %% Create Mobile Terminal (MT) Antenna (2x2 array)
% % Start with an omnidirectional element and replicate it to form a 2x2 array.
% aMT = qd_arrayant('omni');     % Create a single omni element
% aMT.copy_element(1,2);         % Duplicate to have 2 elements
% aMT.copy_element(1,3);         % Third element
% aMT.copy_element(1,4);         % Fourth element
% 
% % Set element positions to form a 2x2 grid (spacing = 0.5 wavelengths)
% % The positions must be given as a 3x4 matrix where each column represents [x; y; z]
% spacing = 0.5;
% aMT.element_position = [ 0,      spacing,  0,      spacing;
%                          0,      0,        spacing, spacing;
%                          0,      0,        0,       0 ];
% aMT.combine_pattern;
% %% Set up the Layout and Scenario
% % Create a new layout with one BS and randomly placed users.
% l = qd_layout;
% l.tx_position = [0; 0; BS_height];    % Place BS at origin, elevated by BS_height
% l.tx_array    = aBS;                  % Assign the BS antenna
% 
% % Generate random user positions (2D Gaussian spread around the BS)
% rng(1);  % for reproducibility
% rx_x = 100 * randn(1, numUsers);       % x-coordinates (in meters)
% rx_y = 100 * randn(1, numUsers);       % y-coordinates (in meters)
% rx_z = MT_height * ones(1, numUsers);  % All users at MT_height
% 
% l.rx_position = [rx_x; rx_y; rx_z];
% l.rx_array    = aMT;  % Assign the 2x2 MT antenna to each user
% 
% % Choose an outdoor scenario – for example, 3GPP UMa (urban macro)
% l.set_scenario('3GPP_38.901_UMa');
% 
% %% Visualize the Environment
% % This command plots the layout, showing BS and user positions.
% l.visualize;
% title('QuaDRiGa Layout Visualization');
% 
% 
% %% Simulation Parameters
% s = qd_simulation_parameters;
% s.center_frequency = carrierFreq;   % Set carrier frequency
% % s.bandwidth = bandwidth;          % Remove or comment this line (bandwidth is not a property)
% s.use_3GPP_baseline = 1;            % Use 3GPP baseline (disable spherical waves)
% s.show_progress_bars = 0;           % Disable progress display
% 
% %% Generate Channel Coefficients
% % Initialize the QuaDRiGa builder and generate channel parameters.
% b = l.init_builder;
% gen_parameters(b);
% c = l.get_channels; % 'c' is a structure array containing the channel coefficients
% 
% % For example, if you wish to extract the channel matrix between the BS and the 1st user:
% % H = c(1, :, 1).coeff; % Channel coefficients for user 1 (modify indexing as needed)
% 
% %% Save the Channel Data
% % Create output directory if it doesn't exist and save the generated channels.
% outputDir = 'output';
% if ~exist(outputDir, 'dir')
%     mkdir(outputDir);
% end
% 
% save(fullfile(outputDir, 'channelData.mat'), 'c', 'l', 's', 'downtiltAngle', 'numUsers');
% 
% disp(['Channel data generated with downtilt = ', num2str(downtiltAngle), '° and saved in ', outputDir]);

%%

% generate_quadriga_channels_main.m
%
% This main script generates channel matrices (H) using QuaDRiGa.
% It creates one base station (8x8 UPA) and 20 user terminals (2x2 arrays)
% for an outdoor 3GPP scenario. In addition to downtilt,
% it configures parameters such as angular spreads, shadow fading, K-Factor, RMS delay spread,
% and spatial consistency using helper functions.
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
% Using the '3gpp-mmw' element type, build an 8x8 array.
% The 5th and 6th parameters (here, 6 and downtiltAngle) denote the reference downtilt and
% the actual downtilt, respectively.
aBS = qd_arrayant('3gpp-mmw', 8, 8, [], 6, config.downtiltAngle, 0.5, 1, 2, 2.5, 2.5);
aBS.combine_pattern;  % Pre-calculate the array pattern

%% Create Mobile Terminal (MT) Antenna (2x2 array)
% Start with an omnidirectional element and replicate it to form a 2x2 array.
aMT = qd_arrayant('omni');   % Create a single omni element
aMT.copy_element(1,2);       % Duplicate to have 2 elements
aMT.copy_element(1,3);       % Third element
aMT.copy_element(1,4);       % Fourth element

% Set element positions to form a 2x2 grid (spacing = 0.5 wavelengths)
% Positions must be a 3x4 matrix (each column = [x; y; z]).
spacing = 0.5;
aMT.element_position = [ 0,      spacing,  0,      spacing;
                         0,      0,        spacing, spacing;
                         0,      0,        0,       0 ];
aMT.combine_pattern;

%% Set up the Layout and Scenario
l = qd_layout;
l.tx_position = [0; 0; config.BS_height];  % BS at origin, elevated by BS_height
l.tx_array    = aBS;                        % Assign BS antenna

% Generate random user positions (2D Gaussian spread around BS)
rng(1);  % For reproducibility
rx_x = 100 * randn(1, config.numUsers);      % x-coordinates [m]
rx_y = 100 * randn(1, config.numUsers);      % y-coordinates [m]
rx_z = config.MT_height * ones(1, config.numUsers); % All users at MT_height
l.rx_position = [rx_x; rx_y; rx_z];
l.rx_array    = aMT;                        % Assign 2x2 MT antenna to each user

% Choose an outdoor scenario – 3GPP UMa (urban macro)
l.set_scenario('3GPP_38.901_UMa');

%% Visualize the Environment
l.visualize;
title('QuaDRiGa Layout Visualization');

%% Simulation Parameters
s = qd_simulation_parameters;
s.center_frequency = config.carrierFreq;  % Set carrier frequency
s.use_3GPP_baseline = 1;                  % Use 3GPP baseline (disable spherical waves)
s.show_progress_bars = 0;                 % Disable progress display
l.simpar = s;

%% Initialize Builder and Configure Additional Parameters
% Initialize the QuaDRiGa builder.
b = l.init_builder;

% Configure additional parameters using helper functions.
b = configure_angles(b, config.perClusterAS_A, config.perClusterAS_D, config.perClusterES_A, config.perClusterES_D);
b = configure_sf_kf_ds(b, config.KF_mu, config.KF_sigma, config.DS_us, config.DS_sigma, config.SF_sigma);
b = configure_spatial_correlation(b, config.SC_lambda);


%% Generate Channel Coefficients
gen_parameters(b);
c = l.get_channels;  % 'c' is a structure array containing the channel coefficients

%% Save the Channel Data
outputDir = 'output';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'channelData.mat'), 'c', 'l', 's', 'config');
disp(['Channel data generated with downtilt = ', num2str(config.downtiltAngle), '° and saved in ', outputDir]);
