import logging
import os
from datetime import datetime

import numpy as np
import seaborn as sns
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.load_data import get_main_dataloader, data_attributes
from models.Evaluator import get_evaluator
from models.GCLOT_multiot_multince import GCLOT
from models.bert import load_pretrained_model_and_tokenizer
from trainer import Trainer
from utils.arguments import args
from utils.path_config import initialize_paths_all, load_models
from utils.util import init_random_state, use_best_hyperparams, cleanup, build_optimizer

# from trainer_neighorx import Trainer

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.set_printoptions(precision=8, sci_mode=False, threshold=1000)
torch.autograd.set_detect_anomaly(False)
import logging

logging.basicConfig(
    level=logging.INFO,  # 控制最低显示级别，DEBUG < INFO < WARNING < ERROR
    format="%(asctime)s | %(levelname)s | %(message)s",  # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        # logging.FileHandler("log.txt")  # 如果你想同时写入文件，可以加这一行
    ]
)


# torch.autograd.set_detect_anomaly(True)


def main(args, trial_params={}):  # , trial

    test_accs = []
    for seed in range(10):
        args.seed = seed
        init_random_state(seed=seed)
        initialize_paths_all(args, seed, fuse_way=args.fuse_way)
        data_attr = data_attributes(args.data_name)
        args.num_nodes = data_attr['num_nodes']
        args.input_dim = data_attr['feat_dim']
        args.num_classes = data_attr['num_classes']
        model = GCLOT(args=args).to(args.device)
        optimizer = build_optimizer(args, model)
        if not load_models(args, model):
            continue  # return 0
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.bert_statedict_path)
        except Exception as e:
            _, tokenizer = load_pretrained_model_and_tokenizer(args.current_dir, args.bert_name, args.num_classes)
        train_loader, bert_infer_loader, gnn_infer_loader, graph, split_idx = get_main_dataloader(args, tokenizer,
            args.device, seed, use_text=True, use_pe=args.use_pe)
        evaluator = get_evaluator(args.data_name, args.num_classes, args.device)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps),
            num_training_steps=total_steps)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: trainable_params:{trainable_params}\n")

        # -------------------------------------- train and test --------------------------------------
        if not os.path.exists(args.best_model_save_path):
            logging.info(f"args.best_model_save_path {args.best_model_save_path} Not Exists.")
        trainer = Trainer(model, optimizer, scheduler, args)
        trainer.fit(graph, evaluator, train_loader, bert_infer_loader, split_idx)

        res = trainer.test(graph, bert_infer_loader, evaluator, split_idx, mode="test")
        metrics = {'acc': res['acc'], 'f1': res['f1'], 'rocauc': res['rocauc']}
        strategy = res['strategy']
        logging.info(
            f" dataset: {args.data_name}, seed: {seed}, {strategy} | acc: {metrics['acc']:.4f}, f1: {metrics['f1']:.4f}, rocauc: {metrics['rocauc']:.4f}")
        acc = res['acc']
        if isinstance(acc, torch.Tensor):
            acc = acc.cpu().item()
        test_accs.append(acc)
        cleanup()

    test_acc_mean = np.mean(test_accs, axis=0)
    std = np.std(test_accs)
    values = np.asarray(test_accs, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    logging.info(
        f'dataset: {args.data_name}, test acc mean ± std = {test_acc_mean:.4f} ± {std:.4f}; test acc mean ± uncertainty = {test_acc_mean:.4f} ± {uncertainty:.4f}')
    print(f'dataset: {args.data_name}, test acc mean ± std = {test_acc_mean:.4f} ± {std:.4f}; test acc mean ± uncertainty = {test_acc_mean:.4f} ± {uncertainty:.4f}')

if __name__ == "__main__":
    logging.info(f"Time:{datetime.now()}, start---------------------------------------------------------")

    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    return_dict = {"result": float("-inf")}
    trial_params = {}
    args.model_name = f"{args.data_name}_pe{args.use_pe}_{args.bert_name}_{args.gnn_name}"
    logging.info(f"args: {args}")
    # print(f"args: {args}")
    main(args)
    cleanup()
