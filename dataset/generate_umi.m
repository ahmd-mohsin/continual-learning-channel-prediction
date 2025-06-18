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
%==========================================================================
%  QuaDRiGa‑based UMi channel generator
%  -------------------------------------------------------------
%  Produces channel tensors of size:
%        [ 500  ×  2  ×  18  ×  8  ×  256 ]
%        (time × Rx‑ant × RB × Tx‑ant × UE)
%--------------------------------------------------------------------------
%  Author :  <your name>        Date : <today>
%==========================================================================

    % ----------------------------- constants -----------------------------
    tx_power            = 1;          % 1 W  ( = 30 dBm )
    no_rx               = 256;        % number of user equipments
    no_time_samples     = 500;        % time dimension
    no_resource_blocks  = 18;         % OFDM RBs
    bandwidth           = 100e6;      % 100 MHz
    center_frequency    = 5e9;        % 5 GHz

    if nargin > 0
        rng(seed,'twister');
        fprintf('UMi: Using random seed %d\n',seed);
    end

    % ------------ three UMi “flavours” with exactly 8 Tx elements --------
    configs = {
        struct( ...
            'name','standard', ...
            'M',2,'N',2,'pol',2, ...      % 2×2 panel, dual‑pol  → 8 Tx el.
            'tilt',30,'spacing',0.5, ...
            'rx_type','dipole','rx_elements',2,'rx_pol','+/-45°', ...
            'scenario','3GPP_38.901_UMi_LOS', ...
            'distance_range',[ 50 100 ], ...
            'tx_height',10,'ue_height',1.5 )

        struct( ...
            'name','dense', ...
            'M',2,'N',2,'pol',2, ...
            'tilt',10,'spacing',0.25, ...
            'rx_type','patch','rx_elements',2,'rx_pol','H/V', ...
            'scenario','5G-ALLSTAR_DenseUrban_LOS', ...
            'distance_range',[ 20 60 ], ...
            'tx_height',6,'ue_height',1.0 )

        struct( ...
            'name','compact', ...
            'M',2,'N',2,'pol',2, ...
            'tilt',0,'spacing',1.0, ...
            'rx_type','cross_pol','rx_elements',2,'rx_pol','+/-45°', ...
            'scenario','3GPP_38.901_UMi_NLOS', ...
            'distance_range',[120 200 ], ...
            'tx_height',15,'ue_height',2.0 )
    };

    % ------------------------- loop over configs -------------------------
    for cIdx = 1 : numel(configs)
        cfg = configs{cIdx};
        fprintf('\n>> Processing "%s" configuration …\n',cfg.name);

        % QuaDRiGa simulation parameters ---------------------------------
        s                       = qd_simulation_parameters;
        s.center_frequency      = center_frequency;
        s.sample_density        = 2.5;
        s.use_absolute_delays   = true;
        s.show_progress_bars    = true;
        if nargin>0
            s.seed = seed + cIdx;
        end

        % Layout ---------------------------------------------------------
        l               = qd_layout(s);
        l.tx_array      = qd_arrayant('3gpp-3d',cfg.M,cfg.N, ...
                                      center_frequency,cfg.pol, ...
                                      cfg.tilt,cfg.spacing);
        no_tx_ant       = l.tx_array.no_elements;   % should be 8

        % ---------- Rx array type & polarisation per configuration ------
        switch cfg.rx_type
            case 'dipole'
                l.rx_array = qd_arrayant('dipole');
                if cfg.rx_elements>1
                    l.rx_array.copy_element(1,2);
                    if strcmp(cfg.rx_pol,'+/-45°')
                        l.rx_array.rotate_pattern( 45,'z',1);
                        l.rx_array.rotate_pattern(-45,'z',2);
                    end
                end
            case 'patch'
                l.rx_array = qd_arrayant('patch');
                if cfg.rx_elements>1
                    l.rx_array.copy_element(1,2);
                    if strcmp(cfg.rx_pol,'H/V')
                        l.rx_array.rotate_pattern( 90,'x',2);
                    end
                end
            otherwise  % 'cross_pol'
                l.rx_array = qd_arrayant('xpol');
        end
        no_rx_ant = l.rx_array.no_elements;   % = 2

        % Tx position, users, tracks ------------------------------------
        l.tx_position = [0 ; 0 ; cfg.tx_height];
        l.no_rx       = no_rx;

        for u = 1:no_rx
            d = diff(cfg.distance_range)*rand + cfg.distance_range(1);
            phi = 2*pi*rand;
            pos = [ d*cos(phi) ; d*sin(phi) ; cfg.ue_height ];

            t           = qd_track('linear',2,phi);
            t.name      = sprintf('UE%03d',u);
            t.initial_position = pos;
            t.interpolate_positions(s.samples_per_meter);

            L           = get_length(t);
            t.movement_profile = [0,no_time_samples ; 0,L]';

            t.scenario  = { cfg.scenario };
            l.rx_track(1,u) = t;        %#ok<*AGROW>
        end

        % ------------------------ channel generation --------------------
        ch = l.get_channels();          % generates array of qd_channel
        scale = sqrt(tx_power);

        % Pre‑allocate output tensor (single precision saves disk)
        channel_matrix = complex( zeros( ...
              no_time_samples , ...                 % 1) time
              no_rx_ant , ...                       % 2) Rx antennas (2)
              no_resource_blocks , ...              % 3) RBs (18)
              no_tx_ant , ...                       % 4) Tx antennas (8)
              no_rx , ...                           % 5) users (256)
              'single') );

        for u = 1:no_rx
            fCh = ch(u).fr(bandwidth,no_resource_blocks);  % [Rx Tx RB T]
            Tmax = min(no_time_samples,size(fCh,4));

            channel_matrix(1:Tmax , : , : , : , u) = ...
                 permute( scale * fCh(:,:,:,1:Tmax) , [4 1 3 2] );
                 % permute to [T Rx RB Tx]
        end

        % ----------------------------- export ---------------------------
        if ~exist('outputs','dir'), mkdir('outputs'); end
        outfile = sprintf('outputs/umi_%s_%dTx_%dRx.mat', ...
                          cfg.name,no_tx_ant,no_rx_ant);
        save(outfile,'channel_matrix','cfg','-v7.3');

        sz = size(channel_matrix);
        fprintf('   saved → %s   size = [%d %d %d %d %d]\n', ...
                outfile, sz);
    end
end
