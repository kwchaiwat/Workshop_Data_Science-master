%%%%%%%%%%%%%%%%%%%%%%%%%% ELM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

clear all;clear all;clc;

dataset = load('Loan.txt');
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
%แบ่ง data สำหรับ xTrain ตั้งแต่ 1-388
xTrain = Xnorm(I(1:388),:);
% แบ่ง data สำหรับ xTest ตั้งแต่ 389-554
xTest = Xnorm(I(389:14),:);
% แบ่ง data สำหรับ tTrain ตั้งแต่ 1-388
tTrain = T(I(1:388),:);
% แบ่ง data สำหรับ tTest ตั้งแต่ 389-554
tTest = T(I(389:end),:);

%Training phase
 dim = size(xTrain,2);
 hidden_node = 1000;
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
  fprintf('Training acc.: %f \n',mean(mean(Index1 == Index2))*100);

Y = output_test;
  % Performance of Testing
  [tmp,Index1] = max(Y,[],2);
  [tmp,Index2] = max(tTest,[],2);
  fprintf('Testing acc.: %f \n',mean(mean(Index1 == Index2))*100);
toc;
