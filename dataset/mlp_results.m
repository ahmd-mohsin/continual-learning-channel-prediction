%% MATLAB Code to Plot Loss and SNR vs. Epoch

% Define epoch numbers from 1 to 50
epochs = 1:50;

% Loss values from the training log (replace these with your values if needed)
loss_values = [2361492.476402, 339652.621161, 110424.090186, 56182.715575, 45918.316410, ...
               38731.255854, 32793.767510, 27545.513559, 53098.587470, 17542.646845, ...
               11774.833879, 10075.462606, 8527.175459, 7072.503837, 5902.446370, ...
               4887.289699, 4139.389916, 3975.473278, 2843.908561, 2421.082428, ...
               1981.153106, 1650.273298, 1354.207867, 1164.736106, 955.337286, ...
               830.990423, 1594.799454, 530.512932, 373.064102, 290.772441, ...
               240.842251, 203.077079, 176.469918, 141.906776, 121.729550, ...
               98.643080, 85.278264, 70.073390, 57.968040, 48.167932, 39.271496, ...
               34.995846, 27.998862, 22.060762, 18.623502, 15.385009, 12.638991, ...
               10.351898, 8.635339, 7.135501];

% SNR values (in dB) from the training log
snr_values = [-54.80, -46.75, -39.59, -39.10, -37.60, ...
              -37.47, -35.90, -34.58, -35.21, -33.24, ...
              -31.79, -30.85, -30.20, -29.05, -27.47, ...
              -28.95, -28.21, -26.81, -25.49, -25.68, ...
              -24.01, -23.19, -21.43, -22.20, -21.25, ...
              -20.05, -20.57, -17.59, -15.43, -15.82, ...
              -15.31, -14.23, -14.18, -13.86, -12.69, ...
              -11.97, -10.46, -10.80, -10.40, -9.47, ...
              -8.95, -9.02, -6.69, -6.43, -6.82, ...
              -5.40, -5.62, -4.96, -4.54, -4.48];

% Create a figure with two subplots
figure;

% Plot Loss vs. Epoch
subplot(2,1,1);
plot(epochs, loss_values, 'b-o', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Loss');
title('Loss vs. Epoch');
grid on;

% Plot SNR vs. Epoch
subplot(2,1,2);
plot(epochs, snr_values, 'r-o', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('SNR (dB)');
title('SNR vs. Epoch');
grid on;


%%
% Example MATLAB code for plotting a bar chart of papers from 2005 to 2020

% Years on the x-axis
years = 2005:2020;

% Approximate number of papers (replace these with your actual data)
papers = [
    100    % 2005
    200    % 2006
    350    % 2007
    600    % 2008
    950    % 2009
    1300   % 2010
    2000   % 2011
    2800   % 2012
    4000   % 2013
    5500   % 2014
    8000   % 2015
    12000  % 2016
    22000  % 2017
    30000  % 2018
    42000  % 2019
    52000  % 2020
];

% Create the figure and plot
figure1= figure;
bar(years, papers, 'FaceColor', [0.4940 0.1840 0.5560]); % a purple shade
xlabel('Year','FontSize',16);
ylabel('Numer of Reinforcement LEarning Papers','FontSize',16);
grid on;
box on;

% (Optional) Format x-axis as integer years (if needed)
set(gca, 'XTick', years);
mkdir results/figs
exportgraphics(figure1, './New Folder/RL.pdf')

% If you'd like a y-axis limit (e.g., to 60000) you can set:
% ylim([0, 60000]);


%%

% Define the dataset names and approximate number of papers
datasets = { ...
    'BIG-Bench', ...
    'Scinstruction', ...
    'DebateQA', ...
    'StrategyQA', ...
    'MR-Ben', ...
    'Franklin', ...
    'Multi-LogicEval', ...
    'MalAlgoQA', ...
    'METAL' ...
};

% Approximate number of papers corresponding to each dataset
papers = [225, 8, 4, 20, 3, 2, 4, 2, 4];

% Create a new figure
figure1=figure;

% Plot the bar chart
bar(papers, 'FaceColor', [0.2, 0.2, 0.8]);  % Adjust bar color if desired

% Customize the x-axis ticks and labels
set(gca, 'XTick', 1:length(datasets), ...
         'XTickLabel', datasets, ...
         'XTickLabelRotation', 45, ...  % Rotate labels for better readability
         'FontSize', 16);                % Set font size for tick labels

% Add axis labels and a title with font size 16
xlabel('Dataset', 'FontSize', 16);
ylabel('Number of Papers', 'FontSize', 16);

% Enable grid for better readability
grid on;
box on;
mkdir results/figs
exportgraphics(figure1, './results/RL_papers.pdf')

