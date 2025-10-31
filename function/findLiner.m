function [coe1,coe2]  = findLiner(IE1,IE2)
    coe1 = polyfit(IE1, IE2, 2);
    coe2 = polyfit(IE2, IE1, 2);
end