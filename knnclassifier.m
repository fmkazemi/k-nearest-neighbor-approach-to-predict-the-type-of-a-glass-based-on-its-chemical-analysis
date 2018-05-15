#Farhad M. Kazemi
function result = knnclassifier(trainx, trainy,test, k)
class= unique(trainy);                               
N= size(test,1);                                     
n1=length(trainy);                                   
if ( n1 < k)                                
   error('You specified more neighbors than existed points.')
end  

%// Use distance 
for i=1:N
newpoint=test(i,:); 
%%%dists = sqrt(sum(bsxfun(@minus, trainx, test(i,:)).^2, 2));Euclidean distance
dists = sum(abs(trainx - ones(n1,1)*test(i,:)),2);  
%dist= sqrt(sum((trainx - ones(n1,1)*test(i,:)).^2,2));
%dists = sqrt(sum(trainx-test(i,:)*ones(m1,m2)).^2, 2);

[d,ind] = sort(dists);
ind_closest = ind(1:k);
x_closest = trainx(ind_closest,:);
x_closest_class=trainy(ind_closest,:);
x_closesthist=hist(x_closest_class,class);
[c, best]= max(x_closesthist);                                

result(i,1)= class(best);
end