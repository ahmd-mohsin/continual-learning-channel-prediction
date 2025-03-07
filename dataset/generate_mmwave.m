function generate_mmwave(seed)
    if nargin > 0
        rng(seed, 'twister');
        disp(['mmWave: Using random seed: ', num2str(seed)]);
    end
    
    % mmWave scenario parameters
    no_rx = 128;
    no_time_samples = 30;
    no_resource_blocks = 18; % truncated
    bandwidth = 400e6;
    center_frequency = 28e9;
    
    % Define different antenna configurations for mmWave
    configs = {
        % Config 1: ULA (Uniform Linear Array)
        struct('name', 'ULA', 'M', 16, 'N', 1, 'pol', 3, 'tilt', 10, 'spacing', 0.5, ...
               'nested_M', 1, 'nested_N', 1, 'panel_spacing_v', 1, 'panel_spacing_h', 1, ...
               'rx_type', 'dipole', 'rx_elements', 2, 'rx_pol', '+/-45°'),
        
        % Config 2: URA (Uniform Rectangular Array)
        struct('name', 'URA', 'M', 8, 'N', 8, 'pol', 3, 'tilt', 15, 'spacing', 0.5, ...
               'nested_M', 1, 'nested_N', 1, 'panel_spacing_v', 1, 'panel_spacing_h', 1, ...
               'rx_type', 'patch', 'rx_elements', 2, 'rx_pol', 'H/V'),
        
        % Config 3: Nested arrays for beamforming
        struct('name', 'nested', 'M', 4, 'N', 4, 'pol', 3, 'tilt', 10, 'spacing', 0.5, ...
               'nested_M', 2, 'nested_N', 2, 'panel_spacing_v', 2, 'panel_spacing_h', 2, ...
               'rx_type', 'xpol', 'rx_elements', 2, 'rx_pol', '+/-45°')
    };
    
    % Loop through each configuration
    for config_idx = 1:length(configs)
        config = configs{config_idx};
        
        fprintf('mmWave scenario: Processing configuration %s\n', config.name);
        
        s = qd_simulation_parameters;
        s.center_frequency = center_frequency;
        s.use_absolute_delays = 1;
        s.show_progress_bars = 1;
        
        if isfield(s, 'seed')
            s.seed = seed + config_idx;
        end

        l = qd_layout(s);
        
        % Configure transmitter antenna array
        if strcmp(config.name, 'nested')
            % Use 3gpp-mmw pattern for nested arrays
            l.tx_array = qd_arrayant('3gpp-mmw', config.M, config.N, center_frequency, config.pol, ...
                config.tilt, config.spacing, 1, config.nested_M, config.nested_N, ...
                config.panel_spacing_v, config.panel_spacing_h);
        else
            % Use standard array patterns for ULA and URA
            l.tx_array = qd_arrayant('3gpp-3d', config.M, config.N, center_frequency, config.pol, ...
                config.tilt, config.spacing);
        end
        no_tx_ant = l.tx_array.no_elements;
        fprintf('mmWave config %s: Using %d transmit antenna elements\n', config.name, no_tx_ant);
        
        % Configure receiver antenna array
        switch config.rx_type
            case 'dipole'
                l.rx_array = qd_arrayant('dipole');
                if config.rx_elements > 1
                    l.rx_array.copy_element(1, 2);
                    % For +/-45° polarization, rotate the second element
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
            case 'xpol'
                l.rx_array = qd_arrayant('xpol');
        end

        no_rx_ant = l.rx_array.no_elements;
        l.tx_position = [0; 0; 10];
        channel_matrix = complex(zeros(no_time_samples, no_tx_ant, no_resource_blocks, no_rx_ant, no_rx));
        l.no_rx = no_rx;
        
        % Create user tracks
        for rx_idx = 1:no_rx
            distance = 20 + 80 * rand();
            angle = 2 * pi * rand();

            x_pos = distance * cos(angle);
            y_pos = distance * sin(angle);

            t = qd_track('linear', 1.5, angle);
            t.name = ['UE', num2str(rx_idx)];
            t.initial_position = [x_pos; y_pos; 1.5];

            t.interpolate_positions(s.samples_per_meter);

            track_length = get_length(t);
            t.movement_profile = [0, no_time_samples; 0, track_length]';
            
            % Determine LOS/NLOS state based on distance
            if rand() < exp(-distance / 50)
                scenario = '3GPP_38.901_UMi_LOS';
            else
                scenario = '3GPP_38.901_UMi_NLOS';
            end

            t.scenario = {scenario};
            l.rx_track(1, rx_idx) = t;
        end
        
        % Generate channels
        c = l.get_channels();
        
        % Process channels
        for rx_idx = 1:no_rx
            freq_channel = c(rx_idx).fr(bandwidth, no_resource_blocks);
            for time_idx = 1:min(no_time_samples, size(freq_channel, 4))
                for tx_ant = 1:no_tx_ant
                    for rb = 1:no_resource_blocks
                        for rx_ant = 1:no_rx_ant
                            channel_matrix(time_idx, tx_ant, rb, rx_ant, rx_idx) = ...
                                freq_channel(rx_ant, tx_ant, rb, time_idx);
                        end
                    end
                end
            end
        end
        
        % Save to file with descriptive name
        filename = sprintf('outputs/mmwave_%s_conf_%dtx_%drx.mat', config.name, no_tx_ant, no_rx_ant);
        save(filename, 'channel_matrix', 'config');
        fprintf('mmWave dataset saved to %s with dimensions: [%d, %d, %d, %d, %d]\n', ...
            filename, size(channel_matrix, 1), size(channel_matrix, 2), size(channel_matrix, 3), ...
            size(channel_matrix, 4), size(channel_matrix, 5));
    end
end
