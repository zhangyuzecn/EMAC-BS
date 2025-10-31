function result = dominates(cost1, cost2)
    result = all(cost1 <= cost2) && any(cost1 < cost2);
end