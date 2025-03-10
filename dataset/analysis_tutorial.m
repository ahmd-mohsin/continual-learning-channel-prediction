% %% analysis_3gpp_38901_calibration.m
% % -------------------------------------------------------------------------
% % This script assumes you have ALREADY RUN the main "3GPP 38.901 Full Calibration"
% % script that produced the cell array "c{ic,il}" in your workspace.
% % It will compute and plot the channel gain distributions for each scenario,
% % each antenna config (1 or 2), and each frequency.
% % -------------------------------------------------------------------------
% 
% % Make sure the cell array c{ic, il} is in your workspace:
% % c{ic,il} -> size: (no_rx, no_tx*SOMETHING, no_freq)
% % Each entry is a 'qd_channel' object with field .coeff
% 
% % If you haven't run the main script in this same session, you can do:
% %   run('the_main_3gpp_calibration_script.m');
% 
% % 'select_scenario' in the main script is [1, 2, 3] => UMi, UMa, InH
% scenario_names = {'UMi','UMa','InH'};  % l(1,il).name
% % The main code uses select_fequency in {1,2,3,4} => (6 GHz, 30 GHz, 60 GHz, 70 GHz)
% freq_labels = {'6 GHz','30 GHz','60 GHz','70 GHz'};
% line_colors = {'b','r','m','k'};  % for each freq
% 
% % We'll create one figure per scenario, and in that figure two subplots for config=1 & 2
% % (If the user only ran a subset of scenarios/frequencies, adapt as needed.)
% figure('Name','Channel Gain Distributions','Position',[100 100 1300 700]);
% 
% for il = select_scenario  % loop over scenario indices
%     for ic = 1:2          % antenna config: 1 or 2
% 
%         % Subplot arrangement: 3 rows (one per scenario), 2 columns (two configs)
%         subplot_index = (il-1)*2 + ic;  
%         subplot(3,2,subplot_index);
%         hold on;
% 
%         % We'll gather channel gains for each freq in a separate set of arrays.
%         legend_handles = [];
% 
%         for iF_idx = 1 : no_freq
%             iF = select_fequency(iF_idx);  % the actual freq index used in main script
%             all_gains_dB = [];            % store the channel gains for all users & sectors
% 
%             % Dimensions of c{ic, il}:
%             %   c{ic,il} is (no_rx x <some number of TX sectors> x no_freq).
%             %   For each element c{ic,il}(ir,it,iF) => qd_channel object
%             %   c{ic,il}(ir,it,iF).coeff => complex matrix of path gains (#delay taps, #Rx ant, etc.)
% 
%             [num_rx, num_tx_sectors, ~] = size(c{ic,il});
% 
%             for ir = 1 : num_rx
%                 for it = 1 : num_tx_sectors
% 
%                     % Retrieve the channel object
%                     chan_obj = c{ic,il}(ir, it, iF_idx);
%                     % If the main script stored them differently, adapt accordingly:
%                     %  e.g., c{ic,il}(ir,it,iF) vs c{ic,il}(ir,it,iF_idx).
% 
%                     % We want the total power across all taps, times, Rx antennas, etc.
%                     % Usually .coeff is dimension (#time snapshots x #Rx-ant x #MPC?),
%                     % but it can vary by scenario. 
%                     % The simplest is to flatten .coeff and sum the power:
% 
%                     H = chan_obj.coeff;         % complex matrix
%                     total_power = sum(abs(H(:)).^2);  % sum of squared magnitude
%                     gain_dB = 10*log10(total_power);
% 
%                     all_gains_dB(end+1) = gain_dB;
% 
%                 end
%             end
% 
%             % Now we have an array of gains for this scenario/config/freq.
%             % We'll plot a CDF for them. For a quick method in MATLAB:
%             sorted_gains = sort(all_gains_dB);
%             cdf_vals = (1:length(sorted_gains)) / length(sorted_gains) * 100;  % in percent
% 
%             % Plot
%             lh = plot(sorted_gains, cdf_vals, '-', 'LineWidth', 2, ...
%                 'Color', line_colors{iF});
%             legend_handles(end+1) = lh;
% 
%         end % iF_idx
% 
%         % Cosmetics
%         grid on; box on;
%         xlabel('Channel Gain [dB]');
%         ylabel('CDF [%]');
%         scenario_name = scenario_names{il};
%         title(sprintf('%s - Config %d', scenario_name, ic));
%         legend(legend_handles, freq_labels(select_fequency), 'Location','best');
%         hold off;
%     end
% end
% 
% sgtitle('Channel Gain Distributions (sum of |H|^2, in dB) across scenarios, configs, freqs');


%% analysis_3gpp_38901_calibration_pdf.m
%
% This script calculates and plots PDFs of channel magnitudes in dB:
%   10 * log10(|h|^2) 
% across each scenario (UMi / UMa / InH),
% each antenna configuration (ic = 1 or 2),
% and each frequency (6 / 30 / 60 / 70 GHz).
%
% Prerequisite:
% You must have already run the 3GPP 38.901 Full Calibration code
% so that "c{ic, il}" is in your workspace.

% Scenario and frequency labels:
scenario_names = {'UMi','UMa','InH'};          % l(1,il).name in your code
freq_labels    = {'6 GHz','30 GHz','60 GHz','70 GHz'};
line_colors    = {'b','r','m','k'};           % color for each frequency
bin_edges      = -350:2:-40;                  % dB range for histogram

% We'll make a figure with 3 rows (scenarios) x 2 cols (configs)
figure('Name','PDF of Channel Magnitude Distributions','Position',[100 100 1300 900]);
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

for il = select_scenario          % il in {1,2,3} => UMi, UMa, InH
    for ic = 1:2                  % antenna config index
        % Create next tile
        tile_index = (il-1)*2 + ic;
        nexttile(tile_index);
        hold on;
        
        % We'll keep track of handles for the legend
        legend_handles = [];
        
        % For each frequency used in the simulation
        for iF_idx = 1 : no_freq
            iF = select_fequency(iF_idx);  % actual freq index
            all_mags_dB = [];
            
            % c{ic,il} has size (no_rx, <some # TX sectors>, no_freq)
            [num_rx, num_tx_sectors, ~] = size(c{ic,il});
            
            % Gather magnitudes from each user (ir) and sector (it)
            for ir = 1 : num_rx
                for it = 1 : num_tx_sectors
                    % Get the channel object
                    chan_obj = c{ic,il}(ir, it, iF_idx);
                    
                    % Extract the complex impulse responses
                    H = chan_obj.coeff;
                    
                    % Flatten H, compute 10*log10(|H|^2)
                    h_mags = abs(H(:));
                    h_mags_dB = 20*log10(h_mags + eps); % or 10*log10(h_mags.^2)
                    
                    % Collect
                    all_mags_dB = [all_mags_dB; h_mags_dB];
                end
            end
            
            % Now we have an array of magnitude-dB samples for this scenario+config+freq
            % We'll plot a histogram with pdf normalization
            % Use 'DisplayStyle','stairs' for a line plot
            h_hist = histogram(all_mags_dB, 'BinEdges', bin_edges, ...
                'Normalization','pdf', 'DisplayStyle','stairs', ...
                'LineWidth',2, 'EdgeColor', line_colors{iF});
            
            % For the legend, we'll store one handle
            legend_handles(end+1) = h_hist;
        end
        
        % Cosmetics
        grid on;
        xlabel('Channel Magnitude [dB] (10 log_{10}|h|^{2})');
        ylabel('PDF');
        scenario_str = scenario_names{il};
        title(sprintf('%s - Config %d', scenario_str, ic));
        legend(legend_handles, freq_labels(select_fequency), 'Location','best');
        hold off;
    end
end

sgtitle('PDF of Channel Magnitude Distributions: 10 log_{10}(|h|^{2})');
