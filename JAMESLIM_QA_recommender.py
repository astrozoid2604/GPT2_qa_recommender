import torch
from torch import tensor
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextDataset, TrainerCallback, is_tensorboard_available, EvalPrediction
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import pandas as pd
from tqdm import tqdm
import time
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
from datasets import load_dataset, DatasetDict



cuda_availability= "cuda" if torch.cuda.is_available() else "cpu"
if cuda_availability=="cuda":
    os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]    = "0" 
    dataset_dir                           = "./dataset"
    #num_train_epochs                      =10 #KIV_SETTING1
    num_train_epochs                      =5 #KIV_SETTING2
    per_device_train_batch_size           =16
    gradient_accumulation_steps           =1
    save_steps                            =1000
    save_total_limit                      =2
    fp16                                  =True
else:
    dataset_dir                           = "/Users/jameslim/Downloads/dataset"
    num_train_epochs                      =10
    per_device_train_batch_size           =2
    gradient_accumulation_steps           =1
    save_steps                            =1000
    save_total_limit                      =2
    fp16                                  =False
    

kaggle_data_url  = "asaniczka/1-3m-linkedin-jobs-and-skills-2024"
zip_path         = dataset_dir + "/1-3m-linkedin-jobs-and-skills-2024.zip"
skills_csv_path  = dataset_dir + "/job_skills.csv"
summary_csv_path = dataset_dir + "/job_summary.csv"
posting_csv_path = dataset_dir + "/linkedin_job_postings.csv"
skills_pq_path   = dataset_dir + "/job_skills.parquet"
summary_pq_path  = dataset_dir + "/job_summary.parquet"
posting_pq_path  = dataset_dir + "/linkedin_job_postings.parquet"
merged_pq_path   = dataset_dir + "/merged.parquet"

token_pt_path    = dataset_dir + "/tokenized_text.pt"
#tensorboard_dir  = dataset_dir + "/GPT2_tensorboard" #KIV_SETTING1
#train_dir        = dataset_dir + "/GPT2_training" #KIV_SETTING1
#model_dir        = dataset_dir + "/GPT2_finetuned_model" #KIV_SETTING1
tensorboard_dir  = dataset_dir + "/GPT2_tensorboard_setting2" #KIV_SETTING2
train_dir        = dataset_dir + "/GPT2_training_setting2" #KIV_SETTING2
model_dir        = dataset_dir + "/GPT2_finetuned_model_setting2" #KIV_SETTING1



def print_time(x):
    title_dict = {1:"Settle data sourcing + Merge all 3 job datasets",
                  2:"Consolidate all columns into 1 column + Add linking words to form a sentence",
                  3:"Load pre-trained GPT2 tokenizer and model",
                  4:"Tokenize and format the merged dataset",
                  5:"Set up fine-tuning arguments + Define data collator for language modeling",
                  6:"Set up TensorBoard callback",
                  7:"Set up Accelerator and Trainer ",
                  8:"Save fine-tuned model",
                 }
    print("\n##########################################################################################")
    print(f"# {x}. {title_dict[x]}")
    print("##########################################################################################")
    print("Current Time:", time.strftime("%H:%M:%S", time.localtime()))


    
##########################################################################################
# 01. Settle data sourcing + Merge all 3 job datasets
##########################################################################################
print_time(1)
file_cond = os.path.exists(dataset_dir)
if file_cond==False: 
    print(f"{dataset_dir} does not exist. Initiating makedirs...")
    os.makedirs(dataset_dir)
else: print(f"Dataset directory {dataset_dir} is found.")

file_cond = os.path.exists(zip_path)
if file_cond==False:
    print(f"{zip_path} does not exist. Initiating download from KaggleAPI...")
    api = KaggleApi()
    api.authenticate() #username="astrozoid2604", key="271d95006019eb47f70fb5a5fe23e5a2")
    api.dataset_download_files(kaggle_data_url, path=dataset_dir, unzip=True)
    print(f"Finished downloading dataset {kaggle_data_url} from KaggleAPI")
else: print(f"Zip filepath {zip_path} is found.")
        
file_cond = os.path.exists(skills_csv_path) and os.path.exists(summary_csv_path) and os.path.exists(posting_csv_path)
if file_cond==False:
    print(f"At least 1 dataset CSV file does not exist. Initiating extraction from downloaded zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print(f"Finished extracting files from {zip_path}")
else: print("All 3 dataset CSVs are found.")

file_cond = os.path.exists(skills_pq_path) and os.path.exists(summary_pq_path) or os.path.exists(posting_pq_path)
if file_cond==False:
    print(f"At least 1 dataset PARQUET file does not exist. Initiating CSV-PARQUET conversion...")
    pd.read_csv(skills_csv_path).to_parquet(skills_pq_path)
    pd.read_csv(summary_csv_path).to_parquet(summary_pq_path)
    pd.read_csv(posting_csv_path).to_parquet(posting_pq_path)
    print(f"Finished converting CSV files to PARQUET files")
else: print("All 3 dataset PARQUETs are found.")

file_cond = os.path.exists(merged_pq_path)
if file_cond==False:
    print(f"Merged dataset PARQUET {merged_pq_path} does not exist. Initiating dataset merging...")
    skill_df   = pd.read_parquet(skills_pq_path)
    summary_df = pd.read_parquet(summary_pq_path)
    posting_df = pd.read_parquet(posting_pq_path)
    merged_df  = pd.merge(skill_df, summary_df, on="job_link", how="inner")
    merged_df  = pd.merge(merged_df, posting_df[['job_link', 'job_title', 'company', 'job_location', 'search_country', 'job_level', 'job_type']], on="job_link", how="inner")
    merged_df  = merged_df.dropna()
    merged_df.to_parquet(merged_pq_path)
    print(f"Finished saving {merged_pq_path}")
    print("Shape of merged_df: ", merged_df.shape)
elif file_cond==True and os.path.exists(token_pt_path)==False:
    print(f"Merged dataset PARQUET {merged_pq_path} is found. Initiating reading merged dataset...")
    merged_df  = pd.read_parquet(merged_pq_path).dropna()
    print(f"Finished loading {merged_pq_path}")
    print("Shape of merged_df: ", merged_df.shape)
else: print(f"Finished loading {merged_pq_path}")


##########################################################################################
# 02. Consolidate all columns into 1 column + Add linking words to form a sentence
##########################################################################################
print_time(2)
def combine_cols(job_title, job_skills, job_level, job_type, job_summary, company, job_location, search_country):
    string = "The job title is " + job_title + ". "
    string += "The required skills are " + job_skills + ". "
    string += "The job level is " + job_type + ". "
    string += "The job type is " + job_type + ". "
    string += "Here is the job summary. " + " ".join(job_summary.split("\n")) + ". "
    string += "The hiring company is " + company + ". "
    string += "The job is located at " + job_location + " in country " + search_country + ". "
    string += "This job posting comes from CS5344GROUP08LINKEDIN dataset."
    return string

file_cond = os.path.exists(token_pt_path)
if file_cond==False:
    combined_df = pd.DataFrame()
    combined_df['Combined'] = merged_df.apply(lambda x: combine_cols(x['job_title'], x['job_skills'], x['job_level'], x['job_type'], x['job_summary'], x['company'], x['job_location'], x['search_country']), axis=1)
    print(f"Finished combining all columns of merged dataset.")

##########################################################################################
# 03. Load pre-trained GPT2 tokenizer and model
##########################################################################################
print_time(3)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(torch.device(cuda_availability))
print(f"Finished loading pre-trained GPT2 model")


##########################################################################################
# 04. Tokenize and format the merged dataset
##########################################################################################
print_time(4)
def tokenizer_with_progress(text_list):
    tokenized_text = []
    for text in tqdm(text_list, desc="GPT2 Tokenizer Progress Bar...", ascii=False, ncols=75):
        tokenized_text += [tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(cuda_availability)]
    return tokenized_text

file_cond = os.path.exists(token_pt_path)
if file_cond==False:
    print(f"Token PARQUET file {token_pt_path} does not exist. Initiating text tokenization from merged dataset...")
    text_list = combined_df['Combined'].tolist()
    tokenized_text = tokenizer_with_progress(text_list)
    torch.save(tokenized_text, token_pt_path)
    tokenized_text = [text.to(cuda_availability) for text in tokenized_text]
    print(f"Finished tokenization.")
else:
    print(f"{token_pt_path} is found.")
    tokenized_text = torch.load(token_pt_path) 
    tokenized_text = [text.to(cuda_availability) for text in tokenized_text]
    print(f"Finished reading {token_pt_path}")

##########################################################################################
# 05. Set up fine-tuning arguments + Define data collator for language modeling
##########################################################################################
print_time(5)
training_args = TrainingArguments(output_dir                 =train_dir,
                                  overwrite_output_dir       =True,
                                  num_train_epochs           =num_train_epochs,
                                  per_device_train_batch_size=per_device_train_batch_size,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  save_steps                 =save_steps,
                                  save_total_limit           =save_total_limit,
                                  prediction_loss_only       =True,
                                  weight_decay               =0.01,
                                  save_strategy              ="steps",
                                  #evaluation_strategy        ="no", #KIV_SETTING1
                                  evaluation_strategy        ="steps", #KIV_SETTING2
                                  #do_eval                    =False,#KIV_SETTING1
                                  #eval_steps                 = 40, #KIV_SETTING2
                                  #logging_steps              = 40, #KIV_SETTING2
                                  load_best_model_at_end     = True, #KIV_SETTING2
                                  optim                      ="adamw_torch",
                                  resume_from_checkpoint     =train_dir,
                                  #learning_rate              =5e-5, #KIV_SETTING1
                                  learning_rate              =2e-4, #KIV_SETTING2
                                  logging_strategy           ="steps", 
                                  seed                       =42,
                                  fp16                       =fp16,
                                 )
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print(f"Finished setting up fine-tuning arguments.")


##########################################################################################
# 06. Set up TensorBoard callback
##########################################################################################
print_time(6)
def training_callback_fn(eval_loss, **kwargs):
    """
    Callback function to write training loss into TensorBoard.
    """
    global global_step
    global_step += 1
    writer.add_scalar("training_loss", eval_loss, global_step=global_step)
    
def custom_rewrite_logs(d, mode):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if mode == 'eval' and k.startswith(eval_prefix):
            if k[eval_prefix_len:] == 'loss':
                new_d["combined/" + k[eval_prefix_len:]] = v
        elif mode == 'test' and k.startswith(test_prefix):
            if k[test_prefix_len:] == 'loss':
                new_d["combined/" + k[test_prefix_len:]] = v
        elif mode == 'train':
            if k == 'loss':
                new_d["combined/" + k] = v
    return new_d

class CombinedTensorBoardCallback(TrainerCallback):
    def __init__(self, tb_writers=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard: raise RuntimeError("TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX.")
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401
                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:self._SummaryWriter = None
        self.tb_writers = tb_writers
    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writers = dict(train=self._SummaryWriter(log_dir=os.path.join(log_dir, 'train')),
                                   eval=self._SummaryWriter(log_dir=os.path.join(log_dir, 'eval')))
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero: return
        log_dir = None
        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writers is None:
            self._init_summary_writer(args, log_dir)

        for k, tbw in self.tb_writers.items():
            tbw.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    tbw.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(tbw, "add_hparams"):
                tbw.add_hparams(args.to_sanitized_dict(), metric_dict={})
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writers is None:
            self._init_summary_writer(args)

        for tbk, tbw in self.tb_writers.items():
            logs_new = custom_rewrite_logs(logs, mode=tbk)
            for k, v in logs_new.items():
                if isinstance(v, (int, float)):
                    tbw.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            tbw.flush()
    def on_train_end(self, args, state, control, **kwargs):
        for tbw in self.tb_writers.values():
            tbw.close()
        self.tb_writers = None
    
writer = SummaryWriter(log_dir=tensorboard_dir)
global_step = 0    # Initialize global step
print(f"Finished setting up TensorBoard callback.")

##########################################################################################
# 07. Set up Accelerator and Trainer 
##########################################################################################
print_time(7)
def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

accelerator = Accelerator()
model, training_args, data_collator = accelerator.prepare(model, training_args, data_collator)

training_size = int(0.998*len(tokenized_text))
ds_train, ds_valid = tokenized_text[:training_size:], tokenized_text[training_size:] #KIV_SETTING2

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    #train_dataset=tokenized_text, #KIV_SETTING1
    train_dataset=ds_train, #KIV_SETTING2
    #eval_dataset=None,  # Pass None for eval_dataset #KIV_SETTING1
    eval_dataset=ds_valid,  #KIV_SETTING2
    compute_metrics=compute_metrics, #KIV_SETTING2
    #callbacks=[training_callback_fn],  # Add the callback function
    callbacks=[CombinedTensorBoardCallback]
)
trainer.train()
print("Finished fine-tuning GPT2 model.")


##########################################################################################
# 08. Save fine-tuned model
##########################################################################################
print_time(8)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    trainer.save_model(model_dir)
    print(f"Finished saving fine-tuned model at {model_dir}.")
