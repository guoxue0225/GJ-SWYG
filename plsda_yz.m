clear;
clc;

% 读取外部验证数据（CSV格式）
new_data = readtable('SG-55-yz.csv');  % 行为样本数，没有标签
new_data = table2array(new_data);  % 转换为矩阵（如果是表格形式）

% 加载保存的PLS-DA模型
load('plsda-zzb-211.mat'); 

% 对外部验证数据进行去均值化（减去训练集的均值）
new_data = new_data - ones(size(new_data, 1), 1) * meanx;  

% 使用训练好的模型对外部数据进行预测
new_predictions = zeros(size(new_data, 1), ClassNum);  % 存储新样本的预测结果

for ii = 1:ClassNum
    betapls = xxx2ttt_all{ii} * betattt_all{ii};  % 每个类别的回归系数
    new_predictions(:, ii) = new_data * betapls;  % 对外部样本进行预测
end

% 计算预测结果
[~, new_estvec] = max(new_predictions, [], 2);  

% 输出预测结果
disp('外部验证的预测结果：');
disp(new_estvec); 
