classdef ChannelConfig
    % ChannelConfig holds user-adjustable parameters for channel generation.
    properties
        downtiltAngle = 120;     % Downtilt angle in degrees
        numUsers = 20;           % Number of user terminals
        BS_height = 25;          % Base station height [m]
        MT_height = 1.5;         % Mobile terminal height [m]
        carrierFreq = 3.5e9;     % Carrier frequency [Hz]
        % Angular spread parameters (in degrees)
        perClusterAS_A = 1;      % Tx per-cluster azimuth spread
        perClusterAS_D = 1;      % Rx per-cluster azimuth spread
        perClusterES_A = 1;      % Tx per-cluster elevation spread
        perClusterES_D = 1;      % Rx per-cluster elevation spread
        % SF, K-factor, Delay Spread parameters
        KF_mu = -5;            % Mean K-Factor [dB]
        KF_sigma = 10;         % K-Factor sigma [dB]
        DS_us = 600;           % Median RMS delay spread in microseconds
        DS_sigma = 0.3;        % DS sigma (log10 scale)
        SF_sigma = 8;          % Shadow fading sigma [dB]
        % Spatial consistency parameter
        SC_lambda = 5;         % Decorrelation distance for SSF [m]
    end
    methods
        function obj = ChannelConfig()
            % Default constructor uses the default property values.
        end
    end
end
