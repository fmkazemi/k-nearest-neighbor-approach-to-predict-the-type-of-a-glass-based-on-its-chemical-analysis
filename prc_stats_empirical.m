#Farhad M. Kazemi
% Computes empirical statistics based on classification output.
% 
% Usage:
%     [TPR, FPR, PPV, AUC, AP] = prc_stats_empirical(targs, dvs)
% 
% Arguments:
%     targs: true class labels (targets)
%     dvs: decision values output by the classifier
% 
% Return values:
%     TPR: true positive rate (recall)
%     FPR: false positive rate
%     PPV: positive predictive value (precision)
%     AUC: area under the ROC curve
%     AP: area under the PR curve (average precision)
% 
% -------------------------------------------------------------------------
function [TPR, FPR, PPV, AUC, AP] = prc_stats_empirical(targs, dvs)
    
    % Check input
    assert(all(size(targs)==size(dvs)));
        
    % Sort decision values and true labels according to decision values
    n = length(dvs);
    [dvs_sorted,idx] = sort(dvs,'ascend'); 
    targs_sorted = targs(idx);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    field1 = 'confusionMat';
if nargin < 2
    value1 = targs;
else
    value1 = confusionmat(targs,dvs);
end

numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));
    
%[TP,TN,FP,FN,sensitivity,specificity,precision,f_score] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   TP(class) = value1(class,class);
   tempMat = value1;
   tempMat(:,class) = []; % remove column
   tempMat(class,:) = []; % remove row
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(value1(:,class))-TP(class);
   FN(class) = sum(value1(class,:))-TP(class);
end

field2 = 'accuracy';  value2 = (TP+TN)/(TP+TN+FP+FN);

for class = 1:numOfClasses

    TPR((class)) = TP(class)/(TP(class)+FN(class));
        FPR(class) = FP(class)/(FP(class)+TN(class));
        PPV(class) = TP(class)/(TP(class)+FP(class));
    
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
    
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Inititalize accumulators
   % TPR = repmat(NaN,1,n+1);
    %FPR = repmat(NaN,1,n+1);
    %PPV = repmat(NaN,1,n+1);
    
    % Now slide the threshold along the decision values (the threshold
    % always lies in between two values; here, the threshold represents the
    % decision value immediately to the right of it)
    %for thr = 1:length(dvs_sorted)+1
        
      %  TP = sum(targs_sorted(thr:end)>0);
     %   FN = sum(targs_sorted(1:thr-1)>0);
       % TN = sum(targs_sorted(1:thr-1)<0);
        %FP = sum(targs_sorted(thr:end)<0);
        
        
        %TPR(thr) = TP/(TP+FN);
        %FPR(thr) = FP/(FP+TN);
        %PPV(thr) = TP/(TP+FP);
   % end
    
    % Compute empirical AUC
    %[tmp,tmp,tmp,AUC] = perfcurve(targs,dvs,1)%'ProcessNaN','addtofalse');
    
    % Compute empirical AP
    AP = abs(trapz(TPR(~isnan(PPV)),PPV(~isnan(PPV))));
    
end
