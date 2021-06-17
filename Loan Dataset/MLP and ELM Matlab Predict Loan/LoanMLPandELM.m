close all;clear all;clc;
% รับข้อมูลเข้ามา
dataset = load('LoanMATLAB.txt');
order = dataset(:,1:1);
x = dataset(:,2:12);
xmax = max(x); %ค่าสูงสุด
xmin = min(x); %ค่าต่ำสุด
% normalize ปรับให้ data ตั้งแต่คอลัม 1-4 อยู่ในช่วง 0-1
Xnorm = (x-xmin)./(xmax-xmin);
% T คือ target 0 1
T = dataset(:,13:14);
% sz คือ size ของ data ทั้งหมด เท่ากับ 554
sz = size(dataset,1);
% I คือ Random permutation สุ่มค่าจำนวน 554 เป็นการสลับค่าไปมาจนถึง 554
I = randperm(sz);


%แบ่ง data สำหรับ xTrain ตั้งแต่ 1-388 70%
xTrain = Xnorm(I(1:388),:);
% แบ่ง data สำหรับ xTest ตั้งแต่ 389-554 30%
xTest = Xnorm(I(389:end),:);
% แบ่ง data สำหรับ tTrain ตั้งแต่ 1-388 70%
tTrain = T(I(1:388),:);
% แบ่ง data สำหรับ tTest ตั้งแต่ 389-554 30%
tTest = T(I(389:end),:);
clear X T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% ELM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
%Training phase
 dim = size(xTrain,2);
 hidden_node = 11;
 input_weight = unifrnd(-1,1,dim,hidden_node);
 bias = unifrnd(-1,1,1,hidden_node);
 hidden_layer = 1./(1+exp(-xTrain*input_weight+repmat(bias,size(xTrain,1),1)));
 output_weight = pinv(hidden_layer)*tTrain;
 output_train = hidden_layer*output_weight;
 
 %Test phase
 hidden_layer = 1./(1+exp(-xTest*input_weight+repmat(bias,size(xTest,1),1)));
 output_test = hidden_layer*output_weight;
 
 error_of_ELM =  mse(tTrain-output_train)
 
Y = output_train;
  %Performance of Traning
  [tmp,Index1] = max(Y,[],2);
  [tmp,Index2] = max(tTrain,[],2);
  fprintf('Training 70percent of DATA acc.: %f \n',mean(mean(Index1 == Index2))*100);

Y = output_test;
  % Performance of Testing
  [tmp,Index1] = max(Y,[],2);
  [tmp,Index2] = max(tTest,[],2);
  fprintf('Testing 30percent of DATA acc.: %f \n',mean(mean(Index1 == Index2))*100);
toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% รับข้อมูลเข้ามา
dataset = load('LoneMATLAB.txt');
order = dataset(:,1:1);
x = dataset(:,2:12);
xmax = max(x); %ค่าสูงสุด
xmin = min(x); %ค่าต่ำสุด
% normalize ปรับให้ data ตั้งแต่คอลัม 1-4 อยู่ในช่วง 0-1
Xnorm = (x-xmin)./(xmax-xmin);
% T คือ target 0 1
T = dataset(:,13:14);
% sz คือ size ของ data ทั้งหมด เท่ากับ 554
sz = size(dataset,1);
% I คือ Random permutation สุ่มค่าจำนวน 554 เป็นการสลับค่าไปมาจนถึง 554
I = randperm(sz);


%แบ่ง data สำหรับ xTrain ตั้งแต่ 1-388 70%
xTrain = Xnorm(I(1:388),:);
% แบ่ง data สำหรับ xTest ตั้งแต่ 389-554 30%
xTest = Xnorm(I(389:end),:);
% แบ่ง data สำหรับ tTrain ตั้งแต่ 1-388 70%
tTrain = T(I(1:388),:);
% แบ่ง data สำหรับ tTest ตั้งแต่ 389-554 30%
tTest = T(I(389:end),:);

%%%%%%%%%%%%%%%%%%Model MLP-BP Learning : 1 hidden layer%%%%%%%%%%%%%%%%%%%
clear X T
tic;
 n = 0.01;
 L = 11; %Hidden node
 wi = rands(size(xTrain,2),L);
 bi = rands(1,L);
 wo = rands(L,size(tTrain,2));
 bo = rands(1,size(tTrain,2));
 E = [];
 for k = 1:500
     for i = 1:size(xTrain,1)
         H = logsig(xTrain(i,:)*wi + bi);
         Y = logsig(H*wo + bo);
         
         e = tTrain(i,:) - Y;
         
         dy = e .* Y .* (1-Y);
         dH = H .* (1-H) .* (dy*wo');
         
         wo = wo + n * H'*dy;
         bo = bo + n * dy;
         wi = wi + n * xTrain(i,:)'*dH;
         bi = bi + n * dH;
     end
     H = logsig(xTrain*wi + repmat(bi,size(xTrain,1),1));
     Y = logsig(H*wo + repmat(bo,size(xTrain,1),1));
     E(k) = mse(tTrain - Y);
     plot(E); title('ELM Training and MLP-BP');
     hold on
     test(k) = error_of_ELM;
     plot(test);
     hold off
     xlabel('Iteration (n) '); ylabel('MSE');
     
     drawnow;
 end
 error_of_MLP_BP = E(k)
 %Train Pedic
 H = logsig(xTrain*wi + repmat(bi,size(xTrain,1),1));
 Y = logsig(H*wo + repmat(bo,size(xTrain,1),1));

 %Performance of Traning
 [tmp,Index1] = max(Y,[],2);
 [tmp,Index2] = max(tTrain,[],2);
 fprintf('Training 70percent of DATA acc.: %f \n',mean(mean(Index1 == Index2))*100);
 
 %Testing  Pedic
 H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
 Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
 
 % Performance of Testing
 [tmp,Index1] = max(Y,[],2);
 [tmp,Index2] = max(tTest,[],2);
 fprintf('Testing 30percent of DATA acc.: %f \n',mean(mean(Index1 == Index2))*100);
toc;
