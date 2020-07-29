# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# This script run a cross-validation for deepBN using CEL dataset.

addpath 'src/deepBN/';
addpath 'src/deepBN/dnntoolbox/';
nfolds = 8
data_dir = 'genomes/'
out_dir = 'results/deepBN/'
partition_dir = 'test_partitions/'
dataset = "cel"
if ~isdir('results/')
    mkdir('results/')
end  
if ~isdir('tmp/')
    mkdir('tmp/')
end  
if ~isdir(out_dir)
    mkdir(out_dir)
end
tic
  out_dir


  fprintf('Loading features...')
data = csvread([data_dir, dataset, '.csv'],1,1);
labels = data(:,end);
data = data(:,1:end-1);

% normalization
data(isnan(data))=0;
data(isinf(data))=0;
data = zscore(data);
data(isnan(data))=0;
data(isinf(data))=0;

  fprintf('Done...')

  flog = fopen('tmp/log.log', 'w');
for fold = 0:(nfolds-1)
	 t0 = toc
        
   mirnas = sprintf('%smirnas/%s_fold%d.csv', partition_dir, dataset, fold);
   unlabeled = sprintf('%sunlabeled/%s_fold%d.csv', partition_dir, dataset, fold);
   test_ind = [csvread(mirnas)'; csvread(unlabeled)'];

   % indexes are 0-based
   test_ind = test_ind + 1;
   ind = 1:length(labels);
   train_ind = ind(~ismember(ind,test_ind));
        
   % train
   fprintf("Start training (fold %d from 8)", fold+1)
   [dnnop, trnopts] = dnnoptim(data(train_ind, :), labels(train_ind), false);
   dnn = dnntrain(dnnop,data(train_ind, :), labels(train_ind));
        
   % test.
	 [slabs, scores] = dnnpredict(dnn, data(test_ind,:));
        
csvwrite(sprintf('%s%s_fold%d.csv', out_dir, dataset, fold), scores);
fprintf(flog, '%s, %d, %.3fhs \n', dataset, fold, (toc - t0)/3600);
   
end
fclose(flog)
