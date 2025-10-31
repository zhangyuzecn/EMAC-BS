function fit = fitness(data,ie,ss,position,N_sel)
[~,s_index] = sort(position,'descend');
bands = s_index(1:N_sel);

fit1 = -mean(ie(bands));
fit2 = cssim_rho(bands,ss)/((1+N_sel)*N_sel/2);

fit=[fit1,fit2];
end