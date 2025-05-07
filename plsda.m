clear ;
clc;
load SWYG-FL-660.mat;

samnumv = ones(1, 11) * 60;
ClassNum = 11;

% 提取标签和特征
yyy = yyyout(samnumv);
yyyc = yyy;

xxxc = y;

xxxc = xxxc'; % 转置特征矩阵
[samnum, varnum] = size(xxxc);
meanx = mean(xxxc);
meany = mean(yyyc);

% 特征和标签去均值化
xxxc = xxxc - ones(samnum, 1) * meanx;
yyyc = yyyc - repmat(meany, size(yyyc, 1), 1);

rand('state', 7); % 随机种子

% 随机划分训练集和测试集（7:3比例）
cv = cvpartition(size(yyyc, 1), 'HoldOut', 0.3); % 70%训练，30%测试
trainIndex = cv.training;
testIndex = cv.test;

trainxxx = xxxc(trainIndex, :);
testxxx = xxxc(testIndex, :);
yyyctrain = yyyc(trainIndex, :);
yyyctest = yyyc(testIndex, :);

% 训练集5折交叉验证来选择最佳潜变量
lvm_values = 1:10; % 潜变量数的范围
accurytrain = zeros(length(lvm_values), 1); % 存储每个潜变量数的训练准确率

for lvm = lvm_values
    cv_train = cvpartition(size(yyyctrain, 1), 'KFold', 5); % 5折交叉验证
    foldAccTrain = zeros(cv_train.NumTestSets, 1); % 每折的训练准确率

    for fold = 1:cv_train.NumTestSets
        % 获取训练集和验证集的索引
        trainIndexFold = cv_train.training(fold);
        valIndexFold = cv_train.test(fold);

        % 提取训练集和验证集
        trainxxxFold = trainxxx(trainIndexFold, :);
        valxxxFold = trainxxx(valIndexFold, :);
        yyyctrainFold = yyyctrain(trainIndexFold, :);
        yyycvalFold = yyyctrain(valIndexFold, :);

        % 初始化 yyyplstrainFold 和 yyyplsvalFold 矩阵
        yyyplstrainFold = zeros(size(trainxxxFold, 1), ClassNum);  % 训练集预测结果矩阵
        yyyplsvalFold = zeros(size(valxxxFold, 1), ClassNum);      % 验证集预测结果矩阵

        for ii = 1:ClassNum
            % 对每个类别训练PLS模型
            [betattt, xxx2ttt, ~] = plsdachenxu(trainxxxFold, yyyctrainFold(:, ii), lvm);
            betapls = xxx2ttt * betattt;

            % 训练集和验证集预测
            yyyplstrainFold(:, ii) = trainxxxFold * betapls;  % 对训练集进行预测
            yyyplsvalFold(:, ii) = valxxxFold * betapls;      % 对验证集进行预测
        end

        % 计算验证集准确率
        [~, estvalvec] = max(yyyplsvalFold');
        [~, valvec] = max(yyycvalFold');
        foldAccTrain(fold) = mean(estvalvec == valvec);
    end

    % 计算当前潜变量数的平均训练准确率
    accurytrain(lvm - lvm_values(1) + 1) = mean(foldAccTrain);
end

% 选择最优的潜变量数
[~, bestLvmIdx] = max(accurytrain);

% 将 bestLvmIdx 转换为 lvm_values 数组中的对应值
bestLvm = lvm_values(bestLvmIdx);
disp(['选择的最佳潜变量数: ', num2str(bestLvm)]);

% 在整个训练集上使用最佳潜变量数训练模型
for ii = 1:ClassNum
    [betattt, xxx2ttt, ~] = plsdachenxu(trainxxx, yyyctrain(:, ii), bestLvm);
    betapls = xxx2ttt * betattt;
    yyyplstrain(:, ii) = trainxxx * betapls; % 训练集预测
    yyyplstest(:, ii) = testxxx * betapls; % 测试集预测
    betattt_all{ii} = betattt;  % 保存回归系数
    xxx2ttt_all{ii} = xxx2ttt;  % 保存特征投影矩阵
end

% 计算训练集和测试集的分类结果
[~, estrainvec] = max(yyyplstrain'); % 训练集预测
[~, estestvec] = max(yyyplstest'); % 测试集预测
[~, trainvec] = max(yyyctrain'); % 真实训练标签
[~, testvec] = max(yyyctest'); % 真实测试标签

% 计算训练集准确率
trainAcc = mean(estrainvec == trainvec);
disp(['训练集准确率: ', num2str(trainAcc * 100), '%']);

% 计算测试集准确率
testAcc = mean(estestvec == testvec);
disp(['测试集准确率: ', num2str(testAcc * 100), '%']);

% 绘制混淆矩阵
figure;
confMat = confusionmat(testvec, estestvec); % 计算混淆矩阵
confusionchart(confMat); % 绘制混淆矩阵图

% 输出最终结果
disp('最终的训练和测试结果：');
disp(['最佳潜变量数: ', num2str(bestLvm)]);

% 保存训练模型到.mat文件
save('SG_660_plsda_model.mat', 'betattt_all', 'xxx2ttt_all', 'bestLvm', 'meanx', 'meany', 'ClassNum');
