function [pop1, pop2] = PSO(data, IE, SS, params, N_sel)
%======================================================================
%  Dual-Task Particle Swarm Optimization (PSO)
%
%  Description:
%  This function performs multi-task spectral band selection using a 
%  PSO-based optimization algorithm. Two tasks (datasets) share 
%  knowledge through transfer operations regulated by a random 
%  mating probability (rmp) and transfer strength (TS).
%
%  Inputs:
%    data   - cell array containing two datasets (data{1}, data{2})
%    VC     - cell array of variance contribution values for each task
%    SS     - cell array of spectral similarity matrices for each task
%    params - struct containing algorithm parameters:
%              .rmp, .N_bands, .coe1, .coe2, .N_ind, .N_task, .GX, .TS
%    N_sel  - number of selected bands
%
%  Outputs:
%    pop1   - optimized feature weights for task 1
%    pop2   - optimized feature weights for task 2
%
%  Author: [Your Name]
%  Date:   [Date]
%======================================================================

    %% -----------------------------
    %  Parameter Initialization
    % -----------------------------
    rmp     = params.rmp;         % Random mating probability
    N_bands = params.N_bands;     % Total number of spectral bands
    coe1    = params.coe1;        % Mapping coefficients task2->task1
    coe2    = params.coe2;        % Mapping coefficients task1->task2
    N_ind   = params.N_ind;       % Number of individuals per task
    N_task  = params.N_task;      % Number of tasks
    N_pop   = N_task * N_ind;     % Total population size
    GX      = params.GX;          % Max iterations
    C       = 0.4;                % Cognitive/social factor

    %% -----------------------------
    %  Population Initialization
    % -----------------------------
    X = rand(N_pop, N_bands);          % Position matrix
    V = 0.5 * rand(N_pop, N_bands);    % Velocity matrix

    % Velocity bounds
    v_max = 0.5;
    v_min = -v_max;

    % Personal best initialization
    pbest = X;
    obj_P = zeros(N_pop, 2);  % Fitness placeholder

    %% -----------------------------
    %  Evaluate Initial Population
    % -----------------------------
    for i = 1:N_task
        for j = 1:N_ind
            idx = j + N_ind * (i - 1);
            obj_P(idx, :) = fitness(data{i}, IE{i}, SS{i}, X(idx, :), N_sel);
        end
    end

    obj_pbest = obj_P;

    % Global best initialization (per task)
    idx_gbest1 = findGbest(obj_pbest(1:N_ind, :));
    idx_gbest2 = findGbest(obj_pbest(N_ind+1:end, :));

    gbest(1, :)     = X(idx_gbest1, :);
    gbest(2, :)     = X(N_ind + idx_gbest2, :);
    obj_gbest(1, :) = obj_pbest(idx_gbest1, :);
    obj_gbest(2, :) = obj_pbest(N_ind + idx_gbest2, :);

    % Update rate initialization
    UR = [1, 1];

    %% -----------------------------
    %  Main Optimization Loop
    % -----------------------------
    for t = 1:GX
        N_update = 0;
        N_trans_update = 0;

        % Inertia weight (linearly decreasing)
        W = 0.9 - 0.5 * (t / GX);

        %% --- Iterate over tasks ---
        for i = 1:N_task
            for j = 1:N_ind
                is_transfer = false;
                ind_idx = j + N_ind * (i - 1);

                % Select a parent from another task
                if i == 1
                    parent = pbest(randi([N_ind+1, N_pop]), :);
                    A_rmp = rmp + rmp * UR(1);
                else
                    parent = pbest(randi([1, N_ind]), :);
                    A_rmp = rmp + rmp * UR(2);
                end

                %% Compute transfer strength
                obj_r = fitness(data{i}, IE{i}, SS{i}, parent, N_sel);
                obj_a = mean(obj_P);
                TS = obj_r / obj_a;

                % Disable transfer variants (optional)
                % TS = 1;        % Disable KTI variant
                A_rmp = rmp;     % Disable KTF variant

                %% Velocity Update
                if rand < A_rmp
                    % --- With transfer ---
                    V_new = W * V(ind_idx, :) ...
                          + C * rand * (pbest(ind_idx, :) - X(ind_idx, :)) ...
                          + C * rand * (gbest(i, :) - X(ind_idx, :)) ...
                          + C * TS * rand * (parent - X(ind_idx, :));
                    is_transfer = true;
                else
                    % --- Without transfer ---
                    V_new = W * V(ind_idx, :) ...
                          + C * rand * (pbest(ind_idx, :) - X(ind_idx, :)) ...
                          + C * rand * (gbest(i, :) - X(ind_idx, :));
                end

                %% Velocity Clamping
                V_new = max(min(V_new, v_max), v_min);

                %% Position Update
                X(ind_idx, :) = X(ind_idx, :) + V_new;
                V(ind_idx, :) = V_new;

                % Position boundary handling
                for k = 1:N_bands
                    if X(ind_idx, k) > 1
                        if rand < 0.5
                            X(ind_idx, k) = 1;
                            V(ind_idx, k) = -V(ind_idx, k);
                        else
                            X(ind_idx, k) = rand;
                            V(ind_idx, k) = 0.5 * rand;
                        end
                    elseif X(ind_idx, k) <= 0
                        if rand < 0.5
                            X(ind_idx, k) = 0;
                            V(ind_idx, k) = -V(ind_idx, k);
                        else
                            X(ind_idx, k) = rand;
                            V(ind_idx, k) = 0.5 * rand;
                        end
                    end
                end

                %% Evaluate and Update
                obj_new = fitness(data{i}, IE{i}, SS{i}, X(ind_idx, :), N_sel);

                % Update pbest if improvement
                if dominates(obj_new, obj_pbest(ind_idx, :))
                    obj_pbest(ind_idx, :) = obj_new;
                    pbest(ind_idx, :) = X(ind_idx, :);
                    N_update = N_update + 1;
                    if is_transfer
                        N_trans_update = N_trans_update + 1;
                    end
                end

                % Update gbest if improved
                if dominates(obj_new, obj_gbest(i, :))
                    obj_gbest(i, :) = obj_new;
                    gbest(i, :) = X(ind_idx, :);
                end
            end

            % Update transfer rate
            UR(i) = N_trans_update / (N_update + 1e-3);
        end
    end

    %% -----------------------------
    %  Output Results
    % -----------------------------
    pop1 = gbest(1, :);
    pop2 = gbest(2, :);
end
