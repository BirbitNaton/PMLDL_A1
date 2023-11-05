# Data Exploration
I need to check for outliers and for inapropriate data samples like ones where reference is less toxic than its counterpart, where the toxicity lies on borderlines for either the translation or reference or both.
# Model Exploration 
Check infewrence of my machine by implementing a simple BERT/BART model and trying to train it
### If Low on Resources
Try using a pretrained model to retrain it for detoxification purpose
### If incapable to retrain
Try custom lightweight seq2seq techniques like HMM
# Result 
Managed to retrain a BART model to decently detoxify texts. It ended up 1.5GB, but with good results and not too heavy to train and use. Read Final report for details