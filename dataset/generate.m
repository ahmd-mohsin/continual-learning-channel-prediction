try
    if isempty(gcp('nocreate'))
        parpool('local');
        disp('Parallel processing enabled.');
    end
catch
    warning('Parallel Computing Toolbox not available. Running in serial mode.');
end

seed = 42;
rng(seed);
disp(['Using random seed: ', num2str(seed)]);

generate_uma_scenario = true;
generate_umi_scenario = true;
generate_indoor_scenario = true;
generate_mmwave_scenario = true;

if ~isfolder("outputs/")
    mkdir("outputs/")
end

tic;

if generate_uma_scenario
    disp('Generating Urban Macro-cell (UMa) scenario datasets...');
    generate_uma(seed);
end

if generate_umi_scenario
    disp('Generating Urban Micro-cell (UMi) scenario datasets...');
    generate_umi(seed);
end

if generate_indoor_scenario
    disp('Generating Indoor Office scenario datasets...');
    generate_indoor(seed);
end

if generate_mmwave_scenario
    disp('Generating mmWave scenario datasets...');
    generate_mmwave(seed);
end

totalTime = toc;
disp(['Channel dataset generation complete in ' num2str(totalTime/60) ' minutes.']);