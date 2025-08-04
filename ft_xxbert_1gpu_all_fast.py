import os
import copy
import logging
from functools import partial

import optuna
import pandas as pd
import torch
import torch.nn as nn
import wandb
from alive_progress import alive_bar
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup

from data.load_data import token_dataloader, get_main_data
from models.bert import load_pretrained_model_and_tokenizer_supervised, load_pretrained_model_and_tokenizer
from utils.path_config import generate_save_paths_ftbert
from utils.plots import plot_tsne
from utils.util import init_random_state, get_class_count, load_checkpoint_2, get_wholedataloader4tSNE, \
    save_checkpoint_scheduler, setup_logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'ftlm.log', filemode='a')
os.environ["WANDB_MODE"] = "offline" # wandb sync wandb/offline-run*


def get_all_embeddings_and_predictions(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # input_ids, attention_mask, labels = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings = outputs.hidden_states[-1][:, 0, :]  # 最后一层 [CLS]
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_embeddings.append(embeddings)
            all_logits.append(logits)
            all_preds.append(preds)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_logits, all_preds, all_labels


def train_model(model, train_dataloader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for batch in train_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, f1


def evaluate_model(model, dataloader, device, return_logits_labels=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            if return_logits_labels:
                all_logits.append(logits.cpu())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    if return_logits_labels:
        all_logits = torch.cat(all_logits, dim=0)
        return avg_loss, accuracy, f1, all_logits, torch.tensor(all_labels)
    else:
        return avg_loss, accuracy, f1


def objective(trial_params, data_name):
    if data_name is None:
        raise ValueError("data_name is None")

    current_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # randseed = [random.randint(0, 9999) for _ in range(10)]
    randseed = [i for i in range(10)]
    model_name = trial_params['model_name']
    batch_size = trial_params['batch_size']
    max_length = trial_params['max_length']
    num_epochs = trial_params['num_epochs']
    learning_rate = trial_params['learning_rate']
    warmup_ratio = trial_params['warmup_ratio']
    use_pe = trial_params['use_pe']

    wandb.login(key="aa1dda53df6b74ec3df6babb69c99a43e67db74e")  # 只在objective第一次login一次
    scaler = GradScaler()

    test_accs = []
    test_f1s = []

    for seed_idx, seed in enumerate(randseed):
        wandb.init(project=f"finetune_{model_name}_{data_name}_pe", config=trial_params, reinit=True)

        init_random_state(seed)
        paths = generate_save_paths_ftbert(current_dir, data_name, model_name, use_pe, max_length, batch_size, seed,
            learning_rate, warmup_ratio)

        num_classes = get_class_count(data_name)
        model, tokenizer = load_pretrained_model_and_tokenizer_supervised(current_dir, model_name, num_classes)
        model = model.to(device)

        graph, text_list, input_dim, num_classes = get_main_data(data_name, current_dir, seed=seed, use_text=True,
            use_pe=use_pe)
        train_loader, val_loader, test_loader, input_dim, num_classes = token_dataloader(graph, text_list,
            paths['token_file'], tokenizer, max_length, batch_size, input_dim, num_classes)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps)

        start_epoch, _ = load_checkpoint_2(model, optimizer, scheduler, filename=paths['checkpoint_path'])
        start_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        model.config.num_labels = num_classes
        if not os.path.isdir(paths['statedict_path']):
            os.makedirs(paths['statedict_path'])
        print("Saving to:", paths['statedict_path'])
        assert not paths['statedict_path'].endswith('.pt'), "Should be a directory"
        with alive_bar(num_epochs) as bar:
            # logging.info({"train"})
            for epoch in range(start_epoch, num_epochs):
                train_loss, train_acc, train_f1 = train_model(model, train_loader, optimizer, scheduler, device,
                    scaler=scaler)
                val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)
                wandb.log(
                    {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc, "train_f1": train_f1,
                     "val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1, })

                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    save_checkpoint_scheduler(model, optimizer, epoch, train_loss, filename=paths['checkpoint_path'],
                        scheduler=scheduler)
                    # 修正路径：确保 statedict_path 是目录
                    if paths['statedict_path'].endswith('.pth'):
                        paths['statedict_path'] = paths['statedict_path'].replace('.pth', '_model')
                    elif paths['statedict_path'].endswith('.pt'):
                        paths['statedict_path'] = paths['statedict_path'].replace('.pt', '_model')
                    if not os.path.isdir(paths['statedict_path']):
                        os.makedirs(paths['statedict_path'])

                    model.save_pretrained(paths['statedict_path'])
                    tokenizer.save_pretrained(paths['statedict_path'])
                    # model.config.num_labels = num_classes
                    # if not os.path.isdir(paths['statedict_path']):
                    #     os.makedirs(paths['statedict_path'])
                    # print("Saving to:", paths['statedict_path'])
                    # # assert not paths['statedict_path'].endswith('.pt'), "Should be a directory"
                    # model.save_pretrained(paths['statedict_path'])
                    # tokenizer.save_pretrained(paths['statedict_path'])
                    # torch.save(model.state_dict(), os.path.join(paths['statedict_path'], "pytorch_model.bin"))
                    # model.config.to_json_file(os.path.join(paths['statedict_path'], "config.json"))
                    # tokenizer.save_pretrained(paths['statedict_path'])
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break
                bar()

        # 测试阶段
        # logging.info({"test"})
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device)
            # torch.save(model.state_dict(), paths['statedict_path'])
            # model.config.num_labels = num_classes
            # if not os.path.isdir(paths['statedict_path']):
            #     os.makedirs(paths['statedict_path'])
            # print("Saving to:", paths['statedict_path'])
            # # assert not paths['statedict_path'].endswith('.pt'), "Should be a directory"
            # model.save_pretrained(paths['statedict_path'])
            # tokenizer.save_pretrained(paths['statedict_path'])
            # torch.save(model.state_dict(), os.path.join(paths['statedict_path'], "pytorch_model.bin"))
            # model.config.to_json_file(os.path.join(paths['statedict_path'], "config.json"))
            # tokenizer.save_pretrained(paths['statedict_path'])

            logging.info(f"Seed {seed}: Test Accuracy={test_acc:.4f}, Test F1={test_f1:.4f}")
            wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})

            test_accs.append(test_acc)
            test_f1s.append(test_f1)

            if seed_idx == 0:
                all_loader, _, _ = get_wholedataloader4tSNE(graph, text_list, paths['token_file'], tokenizer,
                    max_length, batch_size, input_dim, num_classes)
                all_embeddings, all_logits, all_preds, all_labels = get_all_embeddings_and_predictions(model,
                    all_loader, device)
                torch.save({'embeddings': all_embeddings, 'logits': all_logits, 'predictions': all_preds,
                            'labels': all_labels}, paths['embedding_logits_path'])
                plot_tsne(all_embeddings, graph.y, paths['embedding_tsne_path'], title=f"{data_name}_seed{seed}")

        wandb.finish()

    # 汇总结果
    test_accs = torch.tensor(test_accs)
    test_acc_mean = test_accs.mean()
    test_acc_std = test_accs.std()

    logging.info(f"==> {data_name} Final Test Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    wandb.init(project=f"finetune_{model_name}_{data_name}_summary", config=trial_params)
    wandb.log({"Final_Test_Accuracy_Mean": test_acc_mean, "Final_Test_Accuracy_STD": test_acc_std})
    wandb.finish()

    # return test_acc_mean.item()
    return test_acc_mean.item(), test_acc_std.item()


def run_trial(trial, data_name):
    learning_rates = [2e-5]
    trial_params = {'model_name': trial.suggest_categorical('model_name', ['distilbert']),
        'batch_size': trial.suggest_categorical('batch_size', [256]),
        'max_length': trial.suggest_categorical('max_length', [512, 1024]),
        'num_epochs': trial.suggest_categorical('num_epochs', [100]),
        'learning_rate': trial.suggest_categorical('learning_rate', learning_rates),
        'warmup_ratio': trial.suggest_categorical('warmup_ratio', [0.1]),
        'use_pe': trial.suggest_categorical('use_pe', [True, False]), }
    logging.info(f"Trial Params: {trial_params}")
    return objective(trial_params, data_name)


if __name__ == '__main__':
    # python ft_xxbert_1gpu_all_fast.py --datasets texas cornell wisconsin amazon_ratings Actor
    summary_results = []  # 收集 (dataset_name, mean, std)

    setup_logging(log_dir="./", log_filename_prefix="ft_distibert.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  filename=f'res_ftlm.out', filemode='a')
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tuning BERT-based models on graph datasets")

    parser.add_argument('--datasets', nargs='+', default=['cora'],
                        help="List of datasets to process")

    parser.add_argument('--use_optuna', action='store_true', help="Whether to perform hyperparameter tuning")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of trials for Optuna tuning (only if use_optuna)")
    args = parser.parse_args()
    use_optuna = args.use_optuna
    datasets = args.datasets
    n_trials = args.n_trials
    # datasets=["cornell"]
    for data_name in datasets:
        logging.info(f"===== Start processing dataset: {data_name} =====")

        if use_optuna:
            if '__file__' in globals():
                current_dir = os.path.dirname(__file__)
            else:
                current_dir = os.getcwd()
            db_dir = os.path.join(current_dir, "sqlite")
            os.makedirs(db_dir, exist_ok=True)

            db_path = os.path.join(db_dir, f"{data_name}_ft_xxbert.db")

            study = optuna.create_study(direction='maximize', study_name=f"{data_name}_ft_xxbert",
                                        storage=f"sqlite:///{db_path}", load_if_exists=True)
            study.optimize(partial(run_trial, data_name=data_name), n_trials=n_trials, n_jobs=1)

            if study.best_trial is not None:
                logging.info(f"Best hyperparameters for {data_name}: {study.best_params}")
                logging.info(f"Best validation accuracy: {study.best_value:.4f}")
            else:
                logging.warning(f"No successful trials for {data_name}.")
        else:
            # 直接用一组固定默认超参数
            default_params = {
                "model_name": "distilbert",
                "batch_size": 512, # 256
                "max_length": 256, # 512
                "num_epochs": 1000,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "use_pe": True,
            }
            final_test_acc_mean, final_test_acc_std  = objective(default_params, data_name)
            logging.info(f"Dataset: {data_name}, Final Test Accuracy: {final_test_acc_mean*100:.2f}±{final_test_acc_std*100:.2f}")
            summary_results.append((data_name, final_test_acc_mean, final_test_acc_std))

        logging.info(f"===== Finished dataset: {data_name} =====")
        summary_df = pd.DataFrame(summary_results, columns=["Dataset", "Test_Accuracy_Mean", "Test_Accuracy_Std"])
        print(summary_df.to_string(index=False))

        # 保存到文件
        summary_save_path = os.path.join("./", "summary_results.csv")
        summary_df.to_csv(summary_save_path, index=False, mode="a",header=not os.path.isfile(summary_save_path))
        logging.info(f"Summary results saved to {summary_save_path}")