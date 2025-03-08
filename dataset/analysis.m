%% Simplified Channel Analysis Script for UMa Scenario
% This script loads QuaDRiGa-generated channel data for Urban Macro-cell (UMa) scenario
% and plots the channel magnitude histogram and SNR distribution.

clear all;
close all;
clc;

% Create output directory for figures
if ~exist('figures', 'dir')
    mkdir('figures');
end

%% Load the UMa datasets
% Find all UMa datasets
files = dir('outputs/uma_*.mat');
fprintf('Found %d UMa dataset files:\n', length(files));

% Loop through and load each file
configs = {};
channel_stats = struct();

for i = 1:length(files)
    filename = fullfile(files(i).folder, files(i).name);
    fprintf('Loading %s...\n', files(i).name);
    
    % Load the data
    data = load(filename);
    
    % Extract configuration name and store it
    [~, name, ~] = fileparts(files(i).name);
    parts = split(name, '_');
    
    % Handle file naming variations
    if length(parts) >= 3 && strcmp(parts{3}, 'conf')
        % Format: uma_config_conf_*
        config_name = parts{2};
    elseif length(parts) >= 2
        % Format: uma_config_*
        config_name = parts{2};
    else
        % Fallback
        config_name = ['config_', num2str(i)];
    end
    
    % Store the configuration
    configs{i} = config_name;
    fprintf('  Configuration: %s\n', config_name);
    fprintf('  Channel matrix dimensions: [%s]\n', strjoin(string(size(data.channel_matrix)), ', '));
    
    % Store channel matrix
    channel_stats.(config_name).channel_matrix = data.channel_matrix;
    channel_stats.(config_name).config = data.config;
    
    % Calculate channel magnitude over time (averaging over time samples)
    % Use the middle resource block
    middle_rb = ceil(size(data.channel_matrix, 3)/2);
    
    % Average over time and take the middle resource block
    channel_stats.(config_name).time_averaged = squeeze(mean(data.channel_matrix(:,:,middle_rb,:,:), 1));
    
    % Calculate SNR in dB for each user
    % For SNR calculation, we'll use the ratio of average signal power to noise floor
    % We'll assume a noise floor of -100 dBm
    noise_floor_dbm = -100;
    
    % Calculate average channel power for each user
    if ndims(channel_stats.(config_name).time_averaged) >= 3
        user_channel_power = squeeze(mean(mean(abs(channel_stats.(config_name).time_averaged).^2, 1), 2));
    else
        % Handle case with single RX antenna
        user_channel_power = squeeze(mean(abs(channel_stats.(config_name).time_averaged).^2, 1));
    end
    
    % Convert to dBm (assuming transmit power of 0 dBm for simplicity)
    channel_power_dbm = 10*log10(user_channel_power);
    
    % Calculate SNR
    snr_db = channel_power_dbm - noise_floor_dbm;
    channel_stats.(config_name).snr_db = snr_db;
    
    fprintf('  Processing complete.\n');
end

%% Generate Visualizations

% 1. Channel Magnitude Histogram
figure('Position', [100, 100, 900, 500]);
hold on;

config_names = fieldnames(channel_stats);
colors = parula(length(config_names));

for i = 1:length(config_names)
    config = config_names{i};
    
    % Get channel magnitudes from the time-averaged channel
    magnitudes = abs(channel_stats.(config).time_averaged);
    
    % Reshape to get all magnitudes (regardless of dimensions)
    magnitudes_db = 20*log10(magnitudes);
    
    % Calculate histogram with automatic bin selection
    histogram(magnitudes_db, 50, 'Normalization', 'probability', ...
        'DisplayName', config, 'FaceAlpha', 0.6, 'EdgeAlpha', 0.2, ...
        'FaceColor', colors(i,:));
end

grid on;
xlabel('Channel Gain');
ylabel('Probability');
title('Distribution of Channel Gain for Different UMa Configurations');
legend('Location', 'best');
