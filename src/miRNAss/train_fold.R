# Install the package

install.packages("miRNAss", lib="tmp/")
# Load the package
library("miRNAss", lib="tmp/")

args = commandArgs()
data_dir = args[2]
dataset = args[3] 

print("Loading features...")
# Load data
features = read.csv(paste(data_dir, dataset, ".csv",sep=""))
features = subset(features, select=c(-CLASS, -sequence_names))

# Load labels
labels = as.numeric(read.csv("tmp/labels_mirnass.csv",header=FALSE)$V1)
print("Done.")

print("Start training (it may take several hours)...")
pred = miRNAss(features, labels)
print("Done.")

write.csv(pred, "tmp/prediction_mirnass.csv")