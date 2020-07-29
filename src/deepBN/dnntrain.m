function dnn = dnntrain(dnnin,data,labels,trnopts)

labs = [labels 1-labels];
ninp = size(data,2);

if nargin<=3
  trnopts.MaxIter = 32;
  trnopts.BatchSize = 256;
  trnopts.StepRatio = 0.05;
  trnopts.DropOutRate = 0.5;
  trnopts.Object = 'CrossEntropy'; % as used by Thomas
end
trnopts.Verbose = false;

trainIter = trnopts.MaxIter; trnopts.MaxIter = 32;
dnn = pretrainDBN(dnnin, data, trnopts); % basic Hinton RBM, with Gibbs sampling as used by Thomas
trnopts.Layer = 0;
trnopts.MaxIter = trainIter;
dnn = trainDBN(dnn, data, labs, trnopts); % backpr

