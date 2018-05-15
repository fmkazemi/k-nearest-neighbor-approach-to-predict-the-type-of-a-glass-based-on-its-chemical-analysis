#Farhad M. Kazemi
clc
clear
close all
format shortG
%% Insert Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Farhad  M. Kazemi%%%%%%%%%%%%%%%%%%%%%%
load GlassA1data;

X = GlassA1data;
dataRowNumber = size(GlassA1data,1);

crossValidationFolds = 5;
numberOfRowsPerFold = dataRowNumber / crossValidationFolds;

crossValidationTrainData = [];
crossValidationTestData = [];
for startOfRow = 1:numberOfRowsPerFold:dataRowNumber
    testRows = startOfRow:startOfRow+numberOfRowsPerFold-1;
    if (startOfRow == 1)
        trainRows = [max(testRows)+1:dataRowNumber];
        else
        trainRows = [1:startOfRow-1 max(testRows)+1:dataRowNumber];
    end
    
    %crossValidationTrainData = [crossValidationTrainData ; SortedData(trainRows ,:)];
    crossValidationTrainData = [crossValidationTrainData ; X(trainRows ,:)];
    %crossValidationTestData = [crossValidationTestData ;SortedData(testRows ,:)];
    crossValidationTestData = [crossValidationTestData ;X(testRows ,:)];
end
k=[3 25 40];
r=172;
t=43;

result1=knnclassifier(crossValidationTrainData(1:r,1:9),crossValidationTrainData(1:r,10),crossValidationTestData(1:t,1:9), k(1));
%prec_rec(result1, crossValidationTestData(1:t,10));
%%%%%conf1=confusionmatStats(crossValidationTestData(1:t,10),result1);
%Eval1=Evaluate(crossValidationTestData(1:t,10),result1);
%conf11=CopconfusionmatStats(crossValidationTestData(1:t,10),result1);

% Compute empirical curves
[TPR_emp, FPR_emp, PPV_emp] = prc_stats_empirical(crossValidationTestData(1:t,10)', result1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result2=knnclassifier(crossValidationTrainData(r+1:r*2,1:9),crossValidationTrainData(r+1:r*2,10),crossValidationTestData(t+1:t*2,1:9), k(1));
%prec_rec(result2, crossValidationTestData(t+1:t*2,10));
%%conf2=confusionmatStats(crossValidationTestData(t+1:t*2,10),result2);
%Eval2=Evaluate(crossValidationTestData(t+1:t*2,10),result2);
%conf22=CopconfusionmatStats(crossValidationTestData(t+1:t*2,10),result2);

% Compute empirical curves
[TPR_emp2, FPR_emp2, PPV_emp2] = prc_stats_empirical(crossValidationTestData(t+1:t*2,10)', result2');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result3=knnclassifier(crossValidationTrainData(r*2+1:r*3,1:9),crossValidationTrainData(r*2+1:r*3,10),crossValidationTestData(t*2+1:t*3,1:9), k(1));
%prec_rec(result3, crossValidationTestData(t*2+1:t*3,10));
%%conf3=confusionmatStats(crossValidationTestData(t*2+1:t*3,10),result3);
%Eval3=Evaluate(crossValidationTestData(t*2+1:t*3,10),result3);
%%conf33=CopconfusionmatStats(crossValidationTestData(t*2+1:t*3,10),result3);

% Compute empirical curves
[TPR_emp3, FPR_emp3, PPV_emp3] = prc_stats_empirical(crossValidationTestData(t*2+1:t*3,10)', result3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result4=knnclassifier(crossValidationTrainData(r*3+1:r*4,1:9),crossValidationTrainData(r*3+1:r*4,10),crossValidationTestData(t*3+1:t*4,1:9), k(1));
%prec_rec(result4, crossValidationTestData(t*3+1:t*4,10));
%%conf4=confusionmatStats(crossValidationTestData(t*3+1:t*4,10),result4);
%Eval4=Evaluate(crossValidationTestData(t*3+1:t*4,10),result4);
%conf44=CopconfusionmatStats(crossValidationTestData(t*3+1:t*4,10),result4);

% Compute empirical curves
[TPR_emp4, FPR_emp4, PPV_emp4] = prc_stats_empirical(crossValidationTestData(t*3+1:t*4,10)', result4');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result5=knnclassifier(crossValidationTrainData(r*4+1:r*5,1:9),crossValidationTrainData(r*4+1:r*5,10),crossValidationTestData(t*4+1:t*5,1:9), k(1));
%prec_rec(result5, crossValidationTestData(t*4+1:t*5,10));
%%conf5=confusionmatStats(crossValidationTestData(t*4+1:t*5,10),result5);
%Eval5=Evaluate(crossValidationTestData(t*4+1:t*5,10),result5);
%conf55=CopconfusionmatStats(crossValidationTestData(t*4+1:t*5,10),result5);

% Compute empirical curves
[TPR_emp5, FPR_emp5, PPV_emp5] = prc_stats_empirical(crossValidationTestData(t*4+1:t*5,10)', result5');


%recallvector=[conf11.recall conf22.recall conf33.recall conf44.recall conf55.recall];
%precisionvector=[conf11.precision conf22.precision conf33.precision conf44.precision conf55.precision];

recallvector=[TPR_emp TPR_emp2 TPR_emp3 TPR_emp4 TPR_emp5];
precisionvector=[PPV_emp PPV_emp2 PPV_emp3 PPV_emp4 PPV_emp5];
FPRvector=[FPR_emp FPR_emp2 FPR_emp3 FPR_emp4 FPR_emp5];

[FPRvector_sorted indFPR]=sort(FPRvector);
recallvectoradapt=recallvector(indFPR);

[recallvector_sorted indrecall]=sort(recallvector);
precisionvectoradapt=precisionvector(indrecall);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Second k %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result11=knnclassifier(crossValidationTrainData(1:r,1:9),crossValidationTrainData(1:r,10),crossValidationTestData(1:t,1:9), k(2));
%prec_rec(result1, crossValidationTestData(1:t,10));
%%%%%conf1=confusionmatStats(crossValidationTestData(1:t,10),result1);
%Eval1=Evaluate(crossValidationTestData(1:t,10),result1);
%conf11=CopconfusionmatStats(crossValidationTestData(1:t,10),result1);

% Compute empirical curves
[k2TPR_emp, k2FPR_emp, k2PPV_emp] = prc_stats_empirical(crossValidationTestData(1:t,10)', result11');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result22=knnclassifier(crossValidationTrainData(r+1:r*2,1:9),crossValidationTrainData(r+1:r*2,10),crossValidationTestData(t+1:t*2,1:9), k(2));
%prec_rec(result2, crossValidationTestData(t+1:t*2,10));
%%conf2=confusionmatStats(crossValidationTestData(t+1:t*2,10),result2);
%Eval2=Evaluate(crossValidationTestData(t+1:t*2,10),result2);
%conf22=CopconfusionmatStats(crossValidationTestData(t+1:t*2,10),result2);

% Compute empirical curves
[k2TPR_emp2, k2FPR_emp2, k2PPV_emp2] = prc_stats_empirical(crossValidationTestData(t+1:t*2,10)', result22');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result33=knnclassifier(crossValidationTrainData(r*2+1:r*3,1:9),crossValidationTrainData(r*2+1:r*3,10),crossValidationTestData(t*2+1:t*3,1:9), k(2));
%prec_rec(result3, crossValidationTestData(t*2+1:t*3,10));
%%conf3=confusionmatStats(crossValidationTestData(t*2+1:t*3,10),result3);
%Eval3=Evaluate(crossValidationTestData(t*2+1:t*3,10),result3);
%%conf33=CopconfusionmatStats(crossValidationTestData(t*2+1:t*3,10),result3);

% Compute empirical curves
[k2TPR_emp3, k2FPR_emp3, k2PPV_emp3] = prc_stats_empirical(crossValidationTestData(t*2+1:t*3,10)', result33');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result44=knnclassifier(crossValidationTrainData(r*3+1:r*4,1:9),crossValidationTrainData(r*3+1:r*4,10),crossValidationTestData(t*3+1:t*4,1:9), k(2));
%prec_rec(result4, crossValidationTestData(t*3+1:t*4,10));
%%conf4=confusionmatStats(crossValidationTestData(t*3+1:t*4,10),result4);
%Eval4=Evaluate(crossValidationTestData(t*3+1:t*4,10),result4);
%conf44=CopconfusionmatStats(crossValidationTestData(t*3+1:t*4,10),result4);

% Compute empirical curves
[k2TPR_emp4, k2FPR_emp4, k2PPV_emp4] = prc_stats_empirical(crossValidationTestData(t*3+1:t*4,10)', result44');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result55=knnclassifier(crossValidationTrainData(r*4+1:r*5,1:9),crossValidationTrainData(r*4+1:r*5,10),crossValidationTestData(t*4+1:t*5,1:9), k(2));
%prec_rec(result5, crossValidationTestData(t*4+1:t*5,10));
%%conf5=confusionmatStats(crossValidationTestData(t*4+1:t*5,10),result5);
%Eval5=Evaluate(crossValidationTestData(t*4+1:t*5,10),result5);
%conf55=CopconfusionmatStats(crossValidationTestData(t*4+1:t*5,10),result5);

% Compute empirical curves
[k2TPR_emp5, k2FPR_emp5, k2PPV_emp5] = prc_stats_empirical(crossValidationTestData(t*4+1:t*5,10)', result55');


%recallvector=[conf11.recall conf22.recall conf33.recall conf44.recall conf55.recall];
%precisionvector=[conf11.precision conf22.precision conf33.precision conf44.precision conf55.precision];

k2recallvector=[k2TPR_emp k2TPR_emp2 k2TPR_emp3 k2TPR_emp4 k2TPR_emp5];
k2precisionvector=[k2PPV_emp k2PPV_emp2 k2PPV_emp3 k2PPV_emp4 k2PPV_emp5];
k2FPRvector=[k2FPR_emp k2FPR_emp2 k2FPR_emp3 k2FPR_emp4 k2FPR_emp5];

[k2FPRvector_sorted k2indFPR]=sort(k2FPRvector);
k2recallvectoradapt=k2recallvector(k2indFPR);

[k2recallvector_sorted k2indrecall]=sort(k2recallvector);
k2precisionvectoradapt=k2precisionvector(k2indrecall);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Third k %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result111=knnclassifier(crossValidationTrainData(1:r,1:9),crossValidationTrainData(1:r,10),crossValidationTestData(1:t,1:9), k(3));
%prec_rec(result1, crossValidationTestData(1:t,10));
%%%%%conf1=confusionmatStats(crossValidationTestData(1:t,10),result1);
%Eval1=Evaluate(crossValidationTestData(1:t,10),result1);
%conf11=CopconfusionmatStats(crossValidationTestData(1:t,10),result1);

% Compute empirical curves
[k3TPR_emp, k3FPR_emp, k3PPV_emp] = prc_stats_empirical(crossValidationTestData(1:t,10)', result111');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result222=knnclassifier(crossValidationTrainData(r+1:r*2,1:9),crossValidationTrainData(r+1:r*2,10),crossValidationTestData(t+1:t*2,1:9), k(3));
%prec_rec(result2, crossValidationTestData(t+1:t*2,10));
%%conf2=confusionmatStats(crossValidationTestData(t+1:t*2,10),result2);
%Eval2=Evaluate(crossValidationTestData(t+1:t*2,10),result2);
%conf22=CopconfusionmatStats(crossValidationTestData(t+1:t*2,10),result2);

% Compute empirical curves
[k3TPR_emp2, k3FPR_emp2, k3PPV_emp2] = prc_stats_empirical(crossValidationTestData(t+1:t*2,10)', result222');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result333=knnclassifier(crossValidationTrainData(r*2+1:r*3,1:9),crossValidationTrainData(r*2+1:r*3,10),crossValidationTestData(t*2+1:t*3,1:9), k(3));
%prec_rec(result3, crossValidationTestData(t*2+1:t*3,10));
%%conf3=confusionmatStats(crossValidationTestData(t*2+1:t*3,10),result3);
%Eval3=Evaluate(crossValidationTestData(t*2+1:t*3,10),result3);
%%conf33=CopconfusionmatStats(crossValidationTestData(t*2+1:t*3,10),result3);

% Compute empirical curves
[k3TPR_emp3, k3FPR_emp3, k3PPV_emp3] = prc_stats_empirical(crossValidationTestData(t*2+1:t*3,10)', result333');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result444=knnclassifier(crossValidationTrainData(r*3+1:r*4,1:9),crossValidationTrainData(r*3+1:r*4,10),crossValidationTestData(t*3+1:t*4,1:9), k(3));
%prec_rec(result4, crossValidationTestData(t*3+1:t*4,10));
%%conf4=confusionmatStats(crossValidationTestData(t*3+1:t*4,10),result4);
%Eval4=Evaluate(crossValidationTestData(t*3+1:t*4,10),result4);
%conf44=CopconfusionmatStats(crossValidationTestData(t*3+1:t*4,10),result4);

% Compute empirical curves
[k3TPR_emp4, k3FPR_emp4, k3PPV_emp4] = prc_stats_empirical(crossValidationTestData(t*3+1:t*4,10)', result444');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result555=knnclassifier(crossValidationTrainData(r*4+1:r*5,1:9),crossValidationTrainData(r*4+1:r*5,10),crossValidationTestData(t*4+1:t*5,1:9), k(3));
%prec_rec(result5, crossValidationTestData(t*4+1:t*5,10));
%%conf5=confusionmatStats(crossValidationTestData(t*4+1:t*5,10),result5);
%Eval5=Evaluate(crossValidationTestData(t*4+1:t*5,10),result5);
%conf55=CopconfusionmatStats(crossValidationTestData(t*4+1:t*5,10),result5);

% Compute empirical curves
[k3TPR_emp5, k3FPR_emp5, k3PPV_emp5] = prc_stats_empirical(crossValidationTestData(t*4+1:t*5,10)', result555');


%recallvector=[conf11.recall conf22.recall conf33.recall conf44.recall conf55.recall];
%precisionvector=[conf11.precision conf22.precision conf33.precision conf44.precision conf55.precision];

k3recallvector=[k3TPR_emp k3TPR_emp2 k3TPR_emp3 k3TPR_emp4 k3TPR_emp5];
k3precisionvector=[k3PPV_emp k3PPV_emp2 k3PPV_emp3 k3PPV_emp4 k3PPV_emp5];
k3FPRvector=[k3FPR_emp k3FPR_emp2 k3FPR_emp3 k3FPR_emp4 k3FPR_emp5];

[k3FPRvector_sorted k3indFPR]=sort(k3FPRvector);
k3recallvectoradapt=k3recallvector(k3indFPR);

[k3recallvector_sorted k3indrecall]=sort(k3recallvector);
k3precisionvectoradapt=k3precisionvector(k3indrecall);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot results
cols = [200 45 43; 37 64 180; 0 176 80; 0 0 0]/255;

% Plot ROC curves
figure; hold on;
axis([0 1 0 1]); %// Adjust axes for better viewing
grid;
%plot(FPRvector_sorted, recallvectoradapt, '-o',k2FPRvector_sorted, k2recallvectoradapt,'g',k3FPRvector_sorted, k3recallvectoradapt,'c*', 'linewidth', 2);
plot(FPRvector_sorted, recallvectoradapt, '-', 'color', cols(4,:), 'linewidth', 2);
plot(k2FPRvector_sorted, k2recallvectoradapt, '-o', 'color', cols(1,:), 'linewidth', 2);
plot(k3FPRvector_sorted, k3recallvectoradapt, '-', 'color', cols(2,:), 'linewidth', 2);


xlabel('FPR'); ylabel('TPR'); title('ROC curves');
set(gca, 'box', 'on');

% Plot PR(Precision Recall) curves
figure; hold on;
axis([0 1 0 1]); %// Adjust axes for better viewing
grid;
%plot(recallvector_sorted, precisionvectoradapt, '-o',k2recallvector_sorted, k2precisionvectoradapt,'g',k3recallvector_sorted, k3precisionvectoradapt,'c*','linewidth', 2);
plot(recallvector_sorted, precisionvectoradapt, '-', 'color', cols(4,:), 'linewidth', 2);
plot(k2recallvector_sorted, k2precisionvectoradapt, '-o', 'color', cols(1,:), 'linewidth', 2);
plot(k3recallvector_sorted, k3precisionvectoradapt, '-', 'color', cols(2,:), 'linewidth', 2);
xlabel('TPR (recall)'); ylabel('PPV (precision)'); title('PR curves');
legend({'k=3','k=25','k=40'},'FontSize',8,'FontWeight','bold')
set(gca, 'box', 'on');
%%%%%%%%%%%%%%%%%%%%%%%%

