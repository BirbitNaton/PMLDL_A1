import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel
from pandarallel import pandarallel
from pathlib import Path
import warnings
import numpy as np
import evaluate

pandarallel.initialize(progress_bar=True)
warnings.filterwarnings("ignore")

data_dir = Path("data/interim")
models_dir = Path("../models")
formatted_dataset_path = data_dir / "formatted.parquet"
model_path = models_dir / "bart-paraphrase-retrained"
pred_1percent_path = data_dir / "pred_1percent.txt"
true_1percent_path = data_dir / "true_1percent.txt"
target_1percent_path = data_dir / "target_1percent.txt"
stats_path = data_dir / "eval_stats.parquet"

# load data
df = pd.read_parquet(formatted_dataset_path)
index = df.sample(frac=0.01, random_state=42).index
to_predict = df["reference"][index].parallel_apply(lambda x: x.lower()).tolist()
target = df["translation"][index].parallel_apply(lambda x: x.lower()).tolist()

# load model
model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name=model_path)

# predict
predict = model.predict(to_predict)
predict = pd.Series(predict).parallel_apply(lambda x: x.lower()).tolist()

# save results
np.savetxt(pred_1percent_path, predict, fmt='%s')
np.savetxt(true_1percent_path, to_predict, fmt='%s', encoding="utf-8")
np.savetxt(target_1percent_path, target, fmt='%s', encoding="utf-8")

# evaluation model setup
toxicity = evaluate.load("toxicity", module_type="measurement")

# evaluate
pred_toxicity= toxicity.compute(predictions=predict)
pred_toxicity = pred_toxicity["toxicity"]

true_toxicity = toxicity.compute(predictions=to_predict)
true_toxicity = true_toxicity["toxicity"]

target_toxicity = toxicity.compute(predictions=target)
target_toxicity = target_toxicity["toxicity"]

stats = pd.DataFrame({"pred_tox": pred_toxicity,
                      "true_tox": true_toxicity,
                      "target_tox": target_toxicity,
                      "pred_sent": predict,
                      "true_sent": to_predict,
                      "target_sent": target})
stats.to_paruet(stats_path)