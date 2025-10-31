%% ===========================
%  Data Preparation Parameters
%  ===========================
data_type_1 = 'IP';    % Dataset type 1 (e.g., 'IP' for Indian Pines, 'SA' for Salinas)
data_type_2 = 'DI';    % Dataset type 2 (e.g., 'DI' for Dioni)

% Load hyperspectral data and labels
[data1, label1, IE1, SS1] = load_data(data_type_1);
[data2, label2, IE2, SS2] = load_data(data_type_2);

%% ===========================
%  Band Alignment
%  ===========================
[~, idx_sorted] = sort(IE1, 'descend');   
topidx = idx_sorted(177:224);             
data1(:, topidx) = [];
IE1(:, topidx)   = [];
SS1(:, topidx)   = [];
SS1(topidx, :)   = [];

%% ===========================
%  Store Data into Cell Arrays
%  ===========================
data{1}  = data1;
data{2}  = data2;
label{1} = label1;
label{2} = label2;
IE{1}    = IE1;
IE{2}    = IE2;
SS{1}    = SS1;
SS{2}    = SS2;

%% ===========================
%  Algorithm Parameters Setup
%  ===========================
params.N_task   = 2;             % Number of tasks (multi-task optimization)
params.opt_type = 'PSO';         % Optimization algorithm: 'MFEA', 'MFEA_G', or 'PSO'
params.N_bands  = 176;           % Number of selected spectral bands
params.N_ind    = 50;            % Number of individuals (population size)
params.GX       = 100;           % Maximum number of iterations (generations)
params.rmp      = 0.2;           % Random mating probability
params.TS       = 0.4;           % Transfer strength
t_max = 20;  % Number of repetitions for statistical reliability

% Linear mapping coefficients between two datasets
[params.coe1, params.coe2] = findLiner(IE1, IE2);

% Result file naming
resname = [ data_type_2,'_',num2str(t_max),'times_', data_type_1, '&', data_type_2, '_', ...
           params.opt_type, '_fit=IE+SSIM_', ...
           'EachTask', num2str(params.N_ind), 'Individuals_', ...
           num2str(params.GX), 'Iterations_'];

disp(resname);

%% ===========================
%  Band Selection and Classification
%  ===========================


for t = 1:t_max
    for N_sel = 2:30   % Number of selected bands
        tic
        
        % Band selection via PSO optimization
        [pop1, pop2] = PSO(data, IE, SS, params, N_sel);
        tim(t, N_sel) = toc;
        disp(['Search time (s): ', num2str(tim(t, N_sel))]);
        
        % Rank selected bands
        [~, s_index1] = sort(pop1, 'descend');
        [~, s_index2] = sort(pop2, 'descend');
        bands1 = s_index1(1:N_sel);
        bands2 = s_index2(1:N_sel);
        
        % Classification performance evaluation
        [knn_acc_1(t, N_sel), svm_acc_1(t, N_sel), rf_acc_1(t, N_sel), ...
         knn_kappa_1(t, N_sel), svm_kappa_1(t, N_sel), rf_kappa_1(t, N_sel)] = ...
         classify(bands1, data{1}, label{1});
        
        [knn_acc_2(t, N_sel), svm_acc_2(t, N_sel), rf_acc_2(t, N_sel), ...
         knn_kappa_2(t, N_sel), svm_kappa_2(t, N_sel), rf_kappa_2(t, N_sel)] = ...
         classify(bands2, data{2}, label{2});
        
        % Display intermediate results
        disp(['Iteration: ', num2str(t), ...
              ' | Selected bands: ', num2str(N_sel)]);
        
        disp([data_type_1, ':  KNN_acc = ', num2str(knn_acc_1(t, N_sel)), ...
              ' | SVM_acc = ', num2str(svm_acc_1(t, N_sel)), ...
              ' | RF_acc = ', num2str(rf_acc_1(t, N_sel)), ...
              ' | KNN_kappa = ', num2str(knn_kappa_1(t, N_sel)), ...
              ' | SVM_kappa = ', num2str(svm_kappa_1(t, N_sel)), ...
              ' | RF_kappa = ', num2str(rf_kappa_1(t, N_sel))]);
        
        disp([data_type_2, ':  KNN_acc = ', num2str(knn_acc_2(t, N_sel)), ...
              ' | SVM_acc = ', num2str(svm_acc_2(t, N_sel)), ...
              ' | RF_acc = ', num2str(rf_acc_2(t, N_sel)), ...
              ' | KNN_kappa = ', num2str(knn_kappa_2(t, N_sel)), ...
              ' | SVM_kappa = ', num2str(svm_kappa_2(t, N_sel)), ...
              ' | RF_kappa = ', num2str(rf_kappa_2(t, N_sel))]);
    end
end

%% ===========================
%  Save Results for Dataset 1
%  ===========================
filename1 = [data_type_1, '_', resname];

% Compute mean performance
knn_acc_1(t_max+1, :)   = mean(knn_acc_1);
svm_acc_1(t_max+1, :)   = mean(svm_acc_1);
rf_acc_1(t_max+1, :)    = mean(rf_acc_1);
knn_kappa_1(t_max+1, :) = mean(knn_kappa_1);
svm_kappa_1(t_max+1, :) = mean(svm_kappa_1);
rf_kappa_1(t_max+1, :)  = mean(rf_kappa_1);

% Combine timing data
tim = [tim, sum(tim, 2)];
tim(t_max+1, :) = mean(tim);
zero = zeros(t_max+1, 2);

% Save accuracy (OA) results
T_OA = table(100*knn_acc_1, zero, 100*svm_acc_1, zero, 100*rf_acc_1, tim);
filename_OA = ['result/', filename1, '_OA_', datestr(now, 'yyyymmdd'), '.csv'];
writetable(T_OA, filename_OA);

% Save kappa results
T_kappa = table(knn_kappa_1, zero, svm_kappa_1, zero, rf_kappa_1);
filename_KA = ['result/', filename1, '_KP_', datestr(now, 'yyyymmdd'), '.csv'];
writetable(T_kappa, filename_KA);

%% ===========================
%  Save Results for Dataset 2
%  ===========================
filename2 = [data_type_2, '_', resname];

% Compute mean performance
knn_acc_2(t_max+1, :)   = mean(knn_acc_2);
svm_acc_2(t_max+1, :)   = mean(svm_acc_2);
rf_acc_2(t_max+1, :)    = mean(rf_acc_2);
knn_kappa_2(t_max+1, :) = mean(knn_kappa_2);
svm_kappa_2(t_max+1, :) = mean(svm_kappa_2);
rf_kappa_2(t_max+1, :)  = mean(rf_kappa_2);

% Combine timing data
tim = [tim, sum(tim, 2)];
tim(t_max+1, :) = mean(tim);
zero = zeros(t_max+1, 2);

% Save accuracy (OA) results
T_OA = table(100*knn_acc_2, zero, 100*svm_acc_2, zero, 100*rf_acc_2, tim);
filename_OA = ['result/', filename2, '_OA_', datestr(now, 'yyyymmdd'), '.csv'];
writetable(T_OA, filename_OA);

% Save kappa results
T_kappa = table(knn_kappa_2, zero, svm_kappa_2, zero, rf_kappa_2);
filename_KA = ['result/', filename2, '_KP_', datestr(now, 'yyyymmdd'), '.csv'];
writetable(T_kappa, filename_KA);
