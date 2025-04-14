% % Define the hyperparameter values (train steps per iteration)
% trainSteps = [1000, 3000, 5000, 7000, 9000];
% 
% % Define the corresponding evaluation metrics
% Eval_AverageReturn = [4595.396484375, 4762.4384765625, 4614.5869140625, 4808.64501953125, 4449.322265625];
% Eval_StdReturn     = [93.31542205810547, 99.17518615722656, 57.0628776550293, 30.47691535949707, 344.5910949707031];
% 
% % Create a new figure window
% figure;
% 
% % --- Subplot 1: Train steps per iter vs Eval_AverageReturn ---
% subplot(2,1,1);
% plot(trainSteps, Eval_AverageReturn, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('Train Steps per Iteration', 'FontSize', 18);
% ylabel('Eval Average Return', 'FontSize',18);
% title('Train Steps vs. Eval Average Return', 'FontSize', 18);
% grid on;
% 
% % --- Subplot 2: Train steps per iter vs Eval_StdReturn ---
% subplot(2,1,2);
% plot(trainSteps, Eval_StdReturn, 's-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('Train Steps per Iteration', 'FontSize', 18);
% ylabel('Eval StdReturn', 'FontSize', 18);
% title('Train Steps vs. Eval Standard Deviation', 'FontSize', 18);
% grid on;


%%
% Define the hyperparameter values (learning rate)
learningRates = [5e-3, 2.5e-3, 5e-4, 1e-4, 8e-5];

% Define the corresponding evaluation metrics
Eval_AverageReturn = [4595.396484375, 4320.8779296875, -394.97955322265625, -762.2865600585938, -677.7775268554688];
Eval_StdReturn     = [93.31542205810547, 128.9221954345703, 529.8245849609375, 1179.575439453125, 1071.6171875];

% Create a new figure window
figure;

% --- Subplot 1: Learning rate vs. Eval_AverageReturn ---
subplot(2,1,1);
semilogx(learningRates, Eval_AverageReturn, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Learning Rate', 'FontSize', 18);
ylabel('Eval Average Return', 'FontSize', 18);
title('Learning Rate vs. Eval Average Return', 'FontSize', 18);
grid on;

% --- Subplot 2: Learning rate vs. Eval_StdReturn ---
subplot(2,1,2);
semilogx(learningRates, Eval_StdReturn, 's-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Learning Rate', 'FontSize', 18);
ylabel('Eval Standard Deviation', 'FontSize', 18);
title('Learning Rate vs. Eval Std Return', 'FontSize', 18);
grid on;


%%
% Define the number of DAgger iterations (0 to 14)
iterations = 0:14;

% Define the evaluation mean return for each iteration
eval_avg_return = [4595.396484375, ...   % Iteration 0 (BC performance)
                   4725.13916015625, ...  % Iteration 1
                   4568.12890625, ...     % Iteration 2
                   4637.23681640625, ...  % Iteration 3
                   4547.68603515625, ...  % Iteration 4
                   4626.53466796875, ...  % Iteration 5
                   4650.7802734375, ...   % Iteration 6
                   4744.716796875, ...    % Iteration 7
                   4714.7119140625, ...   % Iteration 8
                   4785.3037109375, ...   % Iteration 9
                   4734.2080078125, ...   % Iteration 10
                   4703.25634765625, ...  % Iteration 11
                   4703.8798828125, ...   % Iteration 12
                   4709.52197265625, ...   % Iteration 13
                   4723.68994140625];      % Iteration 14

% Define the corresponding evaluation standard deviation values
eval_std_return = [93.31542205810547, ...  % Iteration 0
                   93.90045166015625, ...  % Iteration 1
                   52.4759635925293, ...   % Iteration 2
                   93.70735168457031, ...  % Iteration 3
                   463.9717712402344, ...  % Iteration 4
                   121.26200103759766, ... % Iteration 5
                   34.05976486206055, ...  % Iteration 6
                   78.21170043945312, ...  % Iteration 7
                   39.61478042602539, ...  % Iteration 8
                   117.0591049194336, ...  % Iteration 9
                   90.29595184326172, ...  % Iteration 10
                   115.85719299316406, ... % Iteration 11
                   74.41100311279297, ...  % Iteration 12
                   108.1612777709961, ...  % Iteration 13
                   75.52738189697266];      % Iteration 14

% Define the expert policy performance (from initial data collection)
expert_return = 4713.6533203125;

% Assume that the behavior cloning (BC) agent performance is given by Iteration 0
bc_return = eval_avg_return(1);

% Create the plot
figure;
errorbar(iterations, eval_avg_return, eval_std_return, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
% Add a horizontal dashed red line for the expert policy's performance
yline(expert_return, 'r--', 'LineWidth', 2);
% Highlight the BC performance at iteration 0 using a green square marker
plot(0, bc_return, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
hold off;

% Label the axes and add title
xlabel('DAgger Iteration', 'FontSize', 12);
ylabel('Evaluation Average Return', 'FontSize', 12);
title('DAgger Learning Curve on Ant-v4', 'FontSize', 14);

% Add a legend
legend('DAgger Mean Return \pm Std', 'Expert Policy', 'Behavioral Cloning (Iteration 0)', 'Location', 'best');

grid on;

%%
% Define the number of DAgger iterations (0 to 14)
iterations = 0:14;

% Define the evaluation mean returns for each iteration (Hopper-v4)
eval_avg_return = [882.81689453125, ...   % Iteration 0 (BC performance)
                   1979.251953125, ...      % Iteration 1
                   1217.13232421875, ...     % Iteration 2
                   2626.790283203125, ...    % Iteration 3
                   1450.207275390625, ...    % Iteration 4
                   3141.64111328125, ...     % Iteration 5
                   3244.951904296875, ...    % Iteration 6
                   3391.049560546875, ...    % Iteration 7
                   3717.41552734375, ...     % Iteration 8
                   3695.26953125, ...        % Iteration 9
                   3690.580078125, ...       % Iteration 10
                   3714.91650390625, ...     % Iteration 11
                   3719.15478515625, ...     % Iteration 12
                   3345.453369140625, ...     % Iteration 13
                   3173.408203125];          % Iteration 14

% Define the evaluation standard deviation for each iteration
eval_std_return = [254.27865600585938, ...    % Iteration 0
                   476.32696533203125, ...    % Iteration 1
                   175.898193359375, ...      % Iteration 2
                   882.1382446289062, ...     % Iteration 3
                   329.63470458984375, ...     % Iteration 4
                   769.488037109375, ...       % Iteration 5
                   794.463623046875, ...      % Iteration 6
                   439.529296875, ...         % Iteration 7
                   4.843890190124512, ...      % Iteration 8
                   37.54642868041992, ...      % Iteration 9
                   8.507781028747559, ...      % Iteration 10
                   7.08574104309082, ...       % Iteration 11
                   5.452010154724121, ...      % Iteration 12
                   806.7338256835938, ...      % Iteration 13
                   783.4978637695312];         % Iteration 14

% Expert policy performance (from initial data collection)
expert_return = 3772.67041015625;

% Behavioral cloning (BC) performance is given by iteration 0
bc_return = eval_avg_return(1);

% Create the plot with error bars
figure;
errorbar(iterations, eval_avg_return, eval_std_return, 'bo-', ...
         'LineWidth', 2, 'MarkerSize', 6);
hold on;
% Add a horizontal dashed line for the expert policy performance
yline(expert_return, 'r--', 'LineWidth', 2);
% Highlight the BC performance at iteration 0 with a green square
plot(0, bc_return, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
hold off;

% Label the axes and add a title
xlabel('DAgger Iteration', 'FontSize', 12);
ylabel('Evaluation Average Return', 'FontSize', 12);
title('DAgger Learning Curve on Hopper-v4', 'FontSize', 14);

% Add a legend to distinguish the curves and markers
legend('DAgger Mean Return \pm Std', 'Expert Policy', ...
       'Behavioral Cloning (Iteration 0)', 'Location', 'best');

grid on;
