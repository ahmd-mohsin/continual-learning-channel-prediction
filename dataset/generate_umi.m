% function generate_umi(seed)
%     if nargin > 0
%         rng(seed, 'twister');
%         disp(['UMi: Using random seed: ', num2str(seed)]);
%     end
% 
%     % UMi scenario parameters
%     no_rx = 256;
%     no_time_samples = 500;
%     no_resource_blocks = 18; % truncated
%     bandwidth = 100e6;
%     center_frequency = 5e9;
% 
%     % Define different antenna configurations for UMi
%     configs = {
%         % Config 1: Standard UMi with +/-45° polarization
%         struct('name', 'standard', 'M', 4, 'N', 2, 'pol', 3, 'tilt', 20, 'spacing', 0.2, ...
%                'rx_type', 'dipole', 'rx_elements', 2, 'rx_pol', '+/-45°'),
% 
%         % Config 2: Dense UMi with H/V polarization
%         struct('name', 'dense', 'M', 6, 'N', 4, 'pol', 5, 'tilt', 10, 'spacing', 0.5, ...
%                'rx_type', 'patch', 'rx_elements', 2, 'rx_pol', 'H/V'),
% 
%         % Config 3: Compact UMi with low downtilt
%         struct('name', 'compact', 'M', 2, 'N', 2, 'pol', 8, 'tilt', 0, 'spacing', 0.8, ...
%                'rx_type', 'cross_pol', 'rx_elements', 2, 'rx_pol', '+/-45°')
%     };
% 
%     % Loop through each configuration
%     for config_idx = 1:length(configs)
%         config = configs{config_idx};
% 
%         fprintf('UMi scenario: Processing configuration %s\n', config.name);
% 
%         s = qd_simulation_parameters;
%         s.center_frequency = center_frequency;
%         s.sample_density = 2.5;
%         s.use_absolute_delays = 1;
%         s.show_progress_bars = 1;
% 
%         if isfield(s, 'seed')
%             s.seed = seed + config_idx;
%         end
% 
%         l = qd_layout(s);
% 
%         % Configure transmitter antenna array
%         l.tx_array = qd_arrayant('3gpp-3d', config.M, config.N, ...
%             center_frequency, config.pol, config.tilt, config.spacing);
%         no_tx_ant = l.tx_array.no_elements;
%         fprintf('UMi config %s: Using %d transmit antenna elements\n', config.name, no_tx_ant);
% 
%         % Configure receiver antenna array
%         switch config.rx_type
%             case 'dipole'
%                 l.rx_array = qd_arrayant('dipole');
%                 if config.rx_elements > 1
%                     l.rx_array.copy_element(1, 2);
%                     if strcmp(config.rx_pol, '+/-45°')
%                         l.rx_array.rotate_pattern(45, 'z', 1);
%                         l.rx_array.rotate_pattern(-45, 'z', 2);
%                     end
%                 end
%             case 'patch'
%                 l.rx_array = qd_arrayant('patch');
%                 if config.rx_elements > 1
%                     l.rx_array.copy_element(1, 2);
%                     if strcmp(config.rx_pol, 'H/V')
%                         l.rx_array.rotate_pattern(90, 'x', 2);
%                     end
%                 end
%             case 'cross_pol'
%                 l.rx_array = qd_arrayant('xpol');
%         end
% 
%         no_rx_ant = l.rx_array.no_elements;
%         l.tx_position = [0; 0; 10];
% 
%         % Prepare storage for channel outputs
%         channel_matrix = complex(zeros(no_time_samples, no_tx_ant, ...
%                                        no_resource_blocks, no_rx_ant, no_rx));
%         l.no_rx = no_rx;
% 
%         % Create user tracks
%         for rx_idx = 1:no_rx
%             distance = 50 + 150 * rand();
%             angle = 2 * pi * rand();
% 
%             x_pos = distance * cos(angle);
%             y_pos = distance * sin(angle);
% 
%             t = qd_track('linear', 2, angle);
%             t.name = ['UE', num2str(rx_idx)];
%             t.initial_position = [x_pos; y_pos; 1.5];
% 
%             t.interpolate_positions(s.samples_per_meter);
% 
%             track_length = get_length(t);
%             t.movement_profile = [0, no_time_samples; 0, track_length]';
% 
%             % Determine LOS/NLOS state based on distance
%             if rand() < exp(-distance / 150)
%                 scenario = '3GPP_38.901_UMi_LOS';
%             else
%                 scenario = '3GPP_38.901_UMi_NLOS';
%             end
% 
%             t.scenario = {scenario};
%             l.rx_track(1, rx_idx) = t;
%         end
% 
%         % Generate channels
%         c = l.get_channels();
% 
%         % Process channels (no TX-power scaling)
%         for rx_idx = 1:no_rx
%             freq_channel = c(rx_idx).fr(bandwidth, no_resource_blocks);
% 
%             for time_idx = 1:min(no_time_samples, size(freq_channel, 4))
%                 for tx_ant = 1:no_tx_ant
%                     for rb = 1:no_resource_blocks
%                         for rx_ant = 1:no_rx_ant
%                             channel_matrix(time_idx, tx_ant, rb, rx_ant, rx_idx) = ...
%                                 freq_channel(rx_ant, tx_ant, rb, time_idx);
%                         end
%                     end
%                 end
%             end
%         end
% 
%         % Save to file with descriptive name
%         filename = sprintf('outputs/umi_%s_conf_%dtx_%drx.mat', ...
%             config.name, no_tx_ant, no_rx_ant);
%         save(filename, 'channel_matrix', 'config', '-v7.3');
%         fprintf('UMi dataset saved to %s with dimensions: [%d, %d, %d, %d, %d]\n', ...
%             filename, size(channel_matrix, 1), size(channel_matrix, 2), ...
%             size(channel_matrix, 3), size(channel_matrix, 4), size(channel_matrix, 5));
%     end
% end


function generate_umi(seed)
    % Transmit power fixed at 1 Watt (do not change)
    tx_power = 1;  % 1 Watt = 30 dBm
    
    if nargin > 0
        rng(seed, 'twister');
        disp(['UMi: Using random seed: ', num2str(seed)]);
    end
    
    % UMi scenario parameters
    no_rx = 256;
    no_time_samples = 500;
    no_resource_blocks = 18; % truncated
    bandwidth = 100e6;
    center_frequency = 5e9;
    
    % ---------------------------------------------------------------------
    % Example: We define three configurations, each using a VALID scenario
    % from the QuaDRiGa v2.8.1 list.
    % ---------------------------------------------------------------------
    configs = {
        % 1) "standard" UMi: moderate distances, forced LOS scenario
        struct('name', 'standard', ...
               'M', 4, 'N', 2, 'pol', 3, ...
               'tilt', 30, 'spacing', 0.5, ...
               'rx_type', 'dipole', 'rx_elements', 2, 'rx_pol', '+/-45°', ...
               'scenario', '3GPP_38.901_UMi_LOS', ...  % valid scenario
               'distance_range', [50, 100], ...
               'tx_height', 10, 'ue_height', 1.5),
        
        % 2) "dense" UMi: small TX height, short distance, forced "dense" LOS
        %    Here we use 5G-ALLSTAR_DenseUrban_LOS as a stand-in for "street-canyon"
        struct('name', 'dense', ...
               'M', 8, 'N', 4, 'pol', 5, ...
               'tilt', 10, 'spacing', 0.25, ...
               'rx_type', 'patch', 'rx_elements', 2, 'rx_pol', 'H/V', ...
               'scenario', '5G-ALLSTAR_DenseUrban_LOS', ...  % valid scenario
               'distance_range', [20, 60], ...
               'tx_height', 6, 'ue_height', 1.0),
        
        % 3) "compact" UMi: larger distance range, purely NLOS
        struct('name', 'compact', ...
               'M', 2, 'N', 2, 'pol', 8, ...
               'tilt', 0, 'spacing', 1.0, ...
               'rx_type', 'cross_pol', 'rx_elements', 2, 'rx_pol', '+/-45°', ...
               'scenario', '3GPP_38.901_UMi_NLOS', ...  % valid scenario
               'distance_range', [120, 200], ...
               'tx_height', 15, 'ue_height', 2.0)
    };
    
    % Loop through each configuration
    for config_idx = 1:length(configs)
        config = configs{config_idx};
        
        fprintf('\nUMi scenario: Processing configuration ''%s''\n', config.name);
        
        % Create simulation parameters
        s = qd_simulation_parameters;
        s.center_frequency = center_frequency;
        s.sample_density = 2.5;
        s.use_absolute_delays = 1;
        s.show_progress_bars = 1;
        
        if isfield(s, 'seed')
            s.seed = seed + config_idx;
        end
        
        % Create layout
        l = qd_layout(s);
        
        % Configure the transmitter antenna array
        l.tx_array = qd_arrayant('3gpp-3d', config.M, config.N, ...
                                 center_frequency, config.pol, ...
                                 config.tilt, config.spacing);
        no_tx_ant = l.tx_array.no_elements;
        
        fprintf('  -> %s config: %d Tx antenna elements\n', ...
                config.name, no_tx_ant);
        
        % Configure the receiver antenna array
        switch config.rx_type
            case 'dipole'
                l.rx_array = qd_arrayant('dipole');
                if config.rx_elements > 1
                    l.rx_array.copy_element(1, 2);
                    if strcmp(config.rx_pol, '+/-45°')
                        l.rx_array.rotate_pattern(45, 'z', 1);
                        l.rx_array.rotate_pattern(-45, 'z', 2);
                    end
                end
                
            case 'patch'
                l.rx_array = qd_arrayant('patch');
                if config.rx_elements > 1
                    l.rx_array.copy_element(1, 2);
                    if strcmp(config.rx_pol, 'H/V')
                        l.rx_array.rotate_pattern(90, 'x', 2);
                    end
                end
                
            case 'cross_pol'
                l.rx_array = qd_arrayant('xpol');
        end
        
        no_rx_ant = l.rx_array.no_elements;
        
        % Set Tx position (height is changed per config)
        l.tx_position = [0; 0; config.tx_height];
        
        % Prepare storage for channel outputs
        channel_matrix = complex(zeros( ...
            no_time_samples, no_tx_ant, no_resource_blocks, no_rx_ant, no_rx));
        l.no_rx = no_rx;
        
        % Create user tracks
        for rx_idx = 1:no_rx
            % Use per-config distance range to produce distinct path losses
            dmin = config.distance_range(1);
            dmax = config.distance_range(2);
            distance = dmin + (dmax - dmin) * rand();
            
            angle = 2 * pi * rand();
            x_pos = distance * cos(angle);
            y_pos = distance * sin(angle);
            
            t = qd_track('linear', 2, angle);
            t.name = ['UE', num2str(rx_idx)];
            t.initial_position = [x_pos; y_pos; config.ue_height];
            
            t.interpolate_positions(s.samples_per_meter);
            
            track_length = get_length(t);
            t.movement_profile = [0, no_time_samples; 0, track_length]';
            
            % Force the scenario from config.scenario
            t.scenario = {config.scenario};
            
            l.rx_track(1, rx_idx) = t;
        end
        
        % Generate channels
        c = l.get_channels();
        
        % Apply TX-power scaling (still 1 W)
        scale_factor = sqrt(tx_power);
        
        % Fill the channel matrix
        for rx_idx = 1:no_rx
            freq_channel = c(rx_idx).fr(bandwidth, no_resource_blocks);
            
            % Make sure not to exceed the time dimension in freq_channel
            for time_idx = 1:min(no_time_samples, size(freq_channel, 4))
                for tx_ant = 1:no_tx_ant
                    for rb = 1:no_resource_blocks
                        for rx_ant = 1:no_rx_ant
                            channel_matrix(time_idx, tx_ant, rb, rx_ant, rx_idx) = ...
                                scale_factor * freq_channel(rx_ant, tx_ant, rb, time_idx);
                        end
                    end
                end
            end
        end
        
        % Save to file with descriptive name
        filename = sprintf('outputs/umi_%s_conf_%dtx_%drx.mat', ...
                           config.name, no_tx_ant, no_rx_ant);
        save(filename, 'channel_matrix', 'config', '-v7.3');
        
        fprintf('  -> Saved: %s  Channel dim: [%d, %d, %d, %d, %d]\n', ...
                filename, size(channel_matrix, 1), size(channel_matrix, 2), ...
                size(channel_matrix, 3), size(channel_matrix, 4), ...
                size(channel_matrix, 5));
    end
end
