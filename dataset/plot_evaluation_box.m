function plot_grouped_bar_comparison()
    % Datasets
    datasets = {'umi\_compact', 'umi\_dense', 'umi\_conference'};

    % Transformer & LSTM Losses
    transformer_losses = [0.0025380376171391127, 1.1872421834129847, 4.175233450829833];
    lstm_losses        = [0.0016176466907367121, 1.219983114882343, 4.167665161655208];

    % Combine data
    data = [transformer_losses; lstm_losses]';

    % Create bar chart
    figure;
    bar_handle = bar(data, 'grouped');
    set(gca, 'XTickLabel', datasets, 'TickLabelInterpreter', 'latex');

    % Colors
    bar_handle(1).FaceColor = [0 0 0];         % Transformer = black
    bar_handle(2).FaceColor = [0.2 0.4 1];     % LSTM = blue-ish

    % Title and axis labels
    title('\textbf{Urban Micro Channel Prediction}', 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('\textbf{Dataset}', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('\textbf{Evaluation Loss}', 'Interpreter', 'latex', 'FontSize', 12);
    grid on;

    % Add value labels on top of bars
    for i = 1:length(datasets)
        text(i - 0.15, transformer_losses(i) + 0.05 * transformer_losses(i), ...
            sprintf('%.4f', transformer_losses(i)), ...
            'HorizontalAlignment', 'center', 'Interpreter', 'latex', 'FontSize', 10);

        text(i + 0.15, lstm_losses(i) + 0.05 * lstm_losses(i), ...
            sprintf('%.4f', lstm_losses(i)), ...
            'HorizontalAlignment', 'center', 'Interpreter', 'latex', 'FontSize', 10);
    end

    legend({'Transformer', 'LSTM'}, 'Interpreter', 'latex', 'Location', 'northwest');
end
