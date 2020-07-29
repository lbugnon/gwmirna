function [rnddnn, trnopts] = dnnoptim(data, labs, optim)
% DNN optimization with the training dataset


ninp = size(data,2);

trnopts.StepRatio = 0.05;
trnopts.DropOutRate = 0.0;
trnopts.Object = 'CrossEntropy';

if optim
    %
    % nodes are 2^nex(i)
    %
    nex = {...
           [3 3 3],...
           [4 3 3], [4 4 3], [4 4 4],...
           [6 4 3], [6 4 4], [6 6 4], [6 6 6],...
           [8 4 3], [8 4 4],...
           [8 6 3], [8 6 4], [8 6 6], ...
           [8 7 4], [8 7 6], [8 7 7], ...
           [8 8 3], [8 8 4], [8 8 6], [8 8 8],...
           [9 3 3],...
           [9 4 3], [9 4 4],...
           [9 6 3], [9 6 4], [9 6 6],...
           [9 8 3], [9 8 4], [9 8 6], [9 8 8],...
           [9 9 3], [9 9 4], [9 9 6], [9 9 8], [9 9 9],...
           [10 10 4], [10 9 6], [10 9 5],...
           [9 8 7 6],...
           [8 8 6 6 4 4],[6 6 6 4],[8 4 6 4 8 4],...
           [6 5 4 6 5 4],[6 4 6 4],[6 8 6 4 6 4],...
           [3 4 3 4 3 4 3 4 3]
          };
    defaultopnodes = [ninp 2^7 2^5 2^4 2]; % simil 100 70 35
    iter = [16 24 32 64 96 128];
    %iter = [200 400 600 1000 1500];

    trnopts.BatchSize = 16;
    trnopts.Object = 'CrossEntropy'; %'Square';

    kf=2;
    [i1trn, i1tst, i0trn, i0tst]=idxxval([labs data],kf);
    rr=zeros(length(nex),length(iter),5,kf);
    for i=1:kf
        trnsel=[i1trn(i,:) i0trn(i,:)];
        data_train=data(trnsel,:);      labs_train=labs(trnsel,:);
        tstsel=[i1tst(i,:) i0tst(i,:)];
        data_test=data(tstsel,:);       labs_test=labs(tstsel,:);
        fprintf('\n');
        
        parfor j=1:length(nex)
            tmpopts = struct(); % def for a parfor limitaton
            tmpopts = trnopts;
            nodop = [ninp];
            for ii=1:length(nex{j}(:)), nodop = [nodop 2^nex{j}(ii)]; end
            nodop = [nodop 2];  disp(nodop);
            rpar = zeros(length(iter),5);
            for k=1:length(iter)
                tmpopts.MaxIter = iter(k);
                dnnin = randDBN(nodop);
                dnn = dnntrain(dnnin,data_train,labs_train,tmpopts);
                slabs = dnnpredict(dnn,data_test);
                rpar(k,:) = claseval(labs_test,slabs) %r=[tpr pre tnr f1 g];
            end
            rr(j,:,:,i) = rpar;
        end
        diary off; diary on; % write log
    end
    
    rr(isnan(rr)) = 0;
    mxkf=max(rr(:,:,4,:),[],4);
    [mx,jkx] = max(mxkf(:));
    [jx,kx]=ind2sub(size(mxkf),jkx);
    if mx > 0
        nodop = [ninp];
        for ii=1:length(nex{jx}(:)), nodop = [nodop 2^nex{jx}(ii)]; end
        nodop = [nodop 2];
        trnopts.MaxIter = iter(kx);
        fprintf('\n%s %6.2f', '*', mx);
    else
        nodop = defaultopnodes;
        trnopts.MaxIter = iter(1);
        fprintf('\n%s ', 'xxxxxxx');
    end
    fprintf('   p =%6d n =', trnopts.MaxIter); disp(nodop);
    fprintf('     ');
    
else % using pre-optimized configurations
    ir = sum(labs==0)/sum(labs==1); %disp(ir);
    if ir<1 +0.5
        nex = {[4 4 3]};
        trnopts.MaxIter = 96;
        trnopts.BatchSize = 32;
    elseif ir<5 +1
        nex = {[4 4 3]};
        trnopts.MaxIter = 32;
        trnopts.BatchSize = 32;
    elseif ir<10 +2
        nex = {[4 4 3]};
        trnopts.MaxIter = 32;
        trnopts.BatchSize = 32;
    elseif ir<50 +10
        nex = {[6 4 4]};
        trnopts.MaxIter = 32;
        trnopts.BatchSize = 32;
    elseif ir<100 +10
        nex = {[8 7 6]};
        trnopts.MaxIter = 32;
        trnopts.BatchSize = 16;
    elseif ir<200 +20
        nex = {[8 7 6]};
        trnopts.MaxIter = 32;
        trnopts.BatchSize = 16;
    elseif ir<500 +50
        nex = {[8 7 4]};
        trnopts.MaxIter = 64;
        trnopts.BatchSize = 16;
    elseif ir<1000 +100
        nex = {[8 7 7]};
        trnopts.MaxIter = 64;
        trnopts.BatchSize = 16;
    elseif ir<1500 +150
        nex = {[8 8 6]};
        trnopts.MaxIter = 64;
        trnopts.BatchSize = 16;
    elseif ir<2000 +200
        nex = {[8 8 6]};
        trnopts.MaxIter = 96;
        trnopts.BatchSize = 16;
    else
        nex = {[8 8 6]};
        trnopts.MaxIter = 128;
        trnopts.BatchSize = 16;
    end
            
    nodop = [ninp];
    for ii=1:length(nex{1}(:)), nodop = [nodop 2^nex{1}(ii)]; end
    nodop = [nodop 2];
end

rnddnn = randDBN(nodop);

