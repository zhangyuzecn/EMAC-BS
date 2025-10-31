function [data,label,IE,SSIM] = load_data(data_type)

if strcmp(data_type, 'IP')
    load Indian_pines.mat;
    load Indian_pines_label.mat;
    data = ones(145, 145, 224);
    data(:,:,2:32) = indian_pines(:,:,1:31);
    data(:,:,34:96) = indian_pines(:,:,32:94);
    data(:,:,98:160) = indian_pines(:,:,95:157);
    data(:,:,162:224) = indian_pines(:,:,158:220);
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'PU')
    load PaviaU.mat;
    load PaviaU_label.mat;
    data(:,:,1:9) = paviaU(:,:,1:9);
    data(:,:,10:102) = paviaU(:,:,11:103);
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'PC')
    load PaviaC.mat;
    load PaviaC_label.mat;
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'SA')
    load Salinas.mat;
    load Salinas_label.mat;
    data = salinas;
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'BW')
    load Botswana.mat;
    load Botswana_gt.mat;
    data = Botswana;
    label = Botswana_gt;
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'KSC')
    load KSC.mat;
    load KSC_gt.mat;
    data = KSC;
    label = KSC_gt; 
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
    
end
if strcmp(data_type, 'HH')
    load WHU_Hi_HongHu.mat;
    load WHU_Hi_HongHu_gt.mat;
    data = WHU_Hi_HongHu;
    label = double(WHU_Hi_HongHu_gt);
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'LK')
    load WHU_Hi_LongKou.mat;
    load WHU_Hi_LongKou_gt.mat;
    data = WHU_Hi_LongKou;
    label = double(WHU_Hi_LongKou_gt);
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end


if strcmp(data_type, 'DI')
    load data/Dioni.mat;
    load data/Dioni_label.mat;
    data = double(data);
    label = double(label);
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end
if strcmp(data_type, 'LO')
    load data/Loukia.mat;
    load data/Loukia_label.mat;
    data = double(data);
    label = double(label);
    [M, N, O] = size(data);       
    IE = computeBandEntropy(reshape(data, M*N, O)');
    SSIM = computeBandSSIM(data);
end



[M,N]=size(label);
label = reshape(label,M*N,1);

label_All = label(:);
label_inds = find(label_All > 0);
label = label(label_inds);

sz = size(data);
data = reshape(data,sz(1)*sz(2),sz(3));
data = double(data(label_inds, :));
data = mapminmax(data,0,1);
data = data +0.0001;



