function [pred,score]= dnnpredict(dnn, data)

out2 = v2h(dnn, data);
pred = max(out2);
[~, idx] = max(out2');
pred=zeros(length(out2),1);
pred(idx==1)=1;
score=out2(:,1);

