import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from pandarallel import pandarallel
from pathlib import Path
import warnings

pandarallel.initialize(progress_bar=True)
warnings.filterwarnings("ignore")

data_dir = Path("data")
models_dir = Path("../models")
formatted_dataset_path = data_dir / "interim/formatted.parquet"
model_path = models_dir / "bart-paraphrase-retrained"

# load data
df = pd.read_parquet(formatted_dataset_path)

df = df.sort_values(by="fit_score", ascending=False).reset_index(drop=True)

# train-val split
train_df = df[["reference", "translation"]].sample(frac=0.8, random_state=42)
eval_df = df[["reference", "translation"]].drop(train_df.index)

eval_df.reset_index(inplace=True, drop=True)
train_df.reset_index(inplace=True, drop=True)
eval_df.columns = ["input_text", "target_text"]
train_df.columns = ["input_text", "target_text"]

# model setup
model_args = Seq2SeqArgs()
model_args.eval_batch_size = 64
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_seq_length = 128
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.train_batch_size = 8
model_args.use_multiprocessing = False
model_args.use_early_stopping = True

model_args.do_sample = True
model_args.num_beams = 1
model_args.num_return_sequences = 1
model_args.max_length = 128
model_args.top_k = 50
model_args.top_p = 0.95
model_args.output_dir = model_path

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="eugenesiow/bart-paraphrase",
    args=model_args,
)

# train
model.train_model(train_df, eval_data=eval_df)