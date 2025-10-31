function [F,R] = fastNonDominatedSort(pop)
    num_obj = size(pop, 2);
    num_ind = size(pop, 1);
    domination_set = cell(1, num_ind);
    dominated_cnt = zeros(1, num_ind);
    F{1} = [];

    for i = 1:num_ind
        for j = i+1:num_ind
            if dominates(pop(i, :), pop(j, :))
                domination_set{i} = [domination_set{i}, j];
                dominated_cnt(j) = dominated_cnt(j) + 1;
            end
            if dominates(pop(j, :), pop(i, :))
                domination_set{j} = [domination_set{j}, i];
                dominated_cnt(i) = dominated_cnt(i) + 1;
            end
        end
        if dominated_cnt(i) == 0
            F{1} = [F{1}, i];
        end
    end

    k = 1;
    while true
        Q = [];
        for i = F{k}
            for j = domination_set{i}
                dominated_cnt(j) = dominated_cnt(j) - 1;
                if dominated_cnt(j) == 0
                    Q = [Q, j];
                end
            end
        end
        if isempty(Q)
            break;
        end
        F{k + 1} = Q;
        k = k + 1;
    end

    R = cell2mat(F);
end

function result = dominates(cost1, cost2)
    result = all(cost1 <= cost2) && any(cost1 < cost2);
end