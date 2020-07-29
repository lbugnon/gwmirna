function [dnnin, trnopts,bestf1] = dnnoptim_randSearch(traindata,bestf1)
% DNN optimization with the training dataset

data=traindata.data;
labs=str2num(cell2mat(traindata.labels));

ninp = size(data,2);

trnopts.Object = 'CrossEntropy';
trnopts.BatchSize = 16;

StepRatio = [0.0001,0.001,0.01,0.05,0.1,0.15];
DropOutRate = [0.0,.2,.5,.7];
Nlayers=[1,2,3];
Nneuron=[3,4,5,6,7,8,9,10];
iter = [16 32 64 96 128];

kf=3;
[i1trn, i1tst, i0trn, i0tst]=idxxval([labs data],kf);

paramsValid={};
f1Valid=[];

while 1
    
    % random parameter set
    trnopts.DropOutRate=datasample(DropOutRate,1);
    trnopts.StepRatio=datasample(StepRatio,1);
    nex=zeros(datasample(Nlayers,1),1);
    for l=1:length(nex)
        nex(l)=datasample(Nneuron,1);
    end
    trnopts.nex=nex;
    trnopts.MaxIter=datasample(iter,1);
    
    nodop = [ninp];
    for ii=1:length(trnopts.nex(:)), nodop = [nodop 2^trnopts.nex(ii)]; end
    nodop = [nodop 2];
    
    dnnin = randDBN(nodop);
    
    rpar=zeros(kf,1);
    
    parfor i=1:kf
        trnsel=[i1trn(i,:) i0trn(i,:)];
        data_train=data(trnsel,:);      labs_train=labs(trnsel,:);
        tstsel=[i1tst(i,:) i0tst(i,:)];
        data_test=data(tstsel,:);       labs_test=labs(tstsel,:);
    
        dnn = dnntrain(dnnin,data_train,labs_train,trnopts);
        
        
        slabs = dnnpredict(dnn,data_test);
        res = claseval(labs_test,slabs); %r=[tpr pre tnr f1 g];
        rpar(i)=res(4);
    end
    
    newf1=mean(rpar);
    
    paramsValid=[paramsValid; trnopts];
    f1Valid=[f1Valid; newf1];
    
    save('log/validRes.m','paramsValid','f1Valid');
    
    if newf1>bestf1
        bestf1=newf1;
        
        return 
    end
    
end

end