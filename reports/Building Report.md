# Data Issues
The data came weirdly processed. First, it was mixed making reference sentences no more toxic than translate. That had to be solved by remapping those with higher toxicity in a row into reference and the other into translated. ![""](figures/Toxicity%20Scores%20Distribution.png)
### What if it's another way around?
Don't really care, if a sample happened to be somehow initially translated so poorly that it became more toxic than it's origin, then let it be as we only care about how one text could become calmer that it was. Meaning, let's learn even on preprocessing mistakes.
![""](figures/Polarized%20Toxicity%20Scores%20Distribution%20[mark(0.1)%20and%20mark(0.9)%20quantiles].png)
### What about borderline cases?
Those hd to be cut as a part of outlier detection. The only use of them that could be would be only using them for training a completely empty model initially incapable of paraphrasing, but as it wasn't the case and as I used a pretrained model to retrain it, I left only most useful scenarios with high toxicity difference.
![""](figures/Polarized%20Toxicity%20Scores%20Distribution%20[cut(0.1)%20and%20mark(0.9)%20quantiles].png)
![""](figures/Polarized%20Toxicity%20Scores%20Distribution%20[cut(0.1)%20and%20cut(0.9)%20quantiles].png)
![""](figures/Difference%20of%20Reference%20and%20Translated%20Scores%20Distribution%20[marked(min)].png)

### What if toxicity drastically reduces in cost of heavy paraphrasing?
With that in mind I took a look into the toxicity difference and similarity and while hadn't a strict dependancy noticed that cutting off a small portion of samples would leave the dataset of both quite similar and quality results.
![""](figures/Correlation%20matrix.png)
![""](figures/Fit%20Score%20Distribution.png)
# Model Selection
While being short on resources, I found ineffective training a model from scratch. I peeked into the [paper provided](https://arxiv.org/abs/2109.08914) in the problem description and saw that authors used conBERT as one of the models researched. With that I looked for a pretrained paraphraser based on BART architecture and decided that it would be wise to rettrain an already working paraphraser to fit the task - detoxification. Before training all it did was *beeping* prophane words, i.e. writing "sh*t" instead of "shit"; it also ignored the majority of latently toxic words and phrases making it sometimes rephrase everything, but a prophane noun or even using it where it shouldn't had been, e.g. processed "you brats killed the gonk!" into "bastards you murdered the gonk!", etc.
# Results
In the end I stopped on choosing a pretrained bart <a href="https://huggingface.co/eugenesiow/bart-paraphrase/blob/main/README.md?code=true" title="paraphraser">paraphraser</a> and retraining it to a decent detoxification capability. The author of the original paraphraser also referred to the <a href="https://github.com/ThilinaRajapakse/simpletransformers/tree/master" title="simpletransformers">simpletransformers</a> open source library as to the interface used for training, so with corresponding time and resources and almost the same pipeline as used to retrained it the same result could be attained.