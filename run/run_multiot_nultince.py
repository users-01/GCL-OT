import os
from datetime import datetime
import logging

import numpy as np
import torch
import seaborn as sns
from transformers import AutoTokenizer

from data.load_data import get_main_dataloader, data_attributes
from models.Evaluator import get_evaluator
from models.OTCL_multiot_multince import OTCL
from models.bert import load_pretrained_model_and_tokenizer
# from trainer import Trainer
from trainer_neighorx import Trainer

from utils.arguments import args
from utils.path_config import initialize_paths_all, load_models
from utils.util import init_random_state, use_best_hyperparams, get_linear_schedule_with_warmup, cleanup, \
    build_optimizer
from wandb_wrapper import setup_wandb

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_printoptions(precision=8, sci_mode=False, threshold=1000)
torch.autograd.set_detect_anomaly(False)

# 配置日志记录器
logging.basicConfig(level=logging.INFO,  # 设置日志记录级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志消息格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置时间格式
    filename='main_cora_otclout_multiple_OT.log', filemode='a'  # 'w' 表示写模式，每次运行都会覆盖文件，'a' 表示追加模式
    )


# torch.autograd.set_detect_anomaly(True)


def main(args, trial_params={}):  # , trial

    # randseed = [random.randint(0, 9999) for _ in range(10)]
    test_accs = []
    for seed in range(10):  # randseed:0, 122, 389, 433, 469, 566, 612, 809
        # seed = 612
        # for seed in [612]:#range(1):  #122, 389, 433, 469, 566, 612, 809
        args.seed = seed
        # logging.info(f'\nseed {seed:02d}:\n args:{args}')
        init_random_state(seed=seed)
        initialize_paths_all(args, seed, fuse_way=args.fuse_way)
        # if args.use_wandb:
        # wandb_run_name = initialize_wandb(args, args.device, seed, str=f"{args.data_name}_{args.gnn_name}_{args.bert_name}_cls3mlp_pe{args.use_pe}")
        wandb = setup_wandb(args, seed,
            suffix=f"{args.data_name}_{args.gnn_name}_{args.bert_name}_cls3mlp_pe{args.use_pe}_{args.fuse_way}")
        data_attr = data_attributes(args.data_name)
        args.num_nodes = data_attr['num_nodes']
        args.input_dim = data_attr['feat_dim']
        args.num_classes = data_attr['num_classes']
        model = OTCL(args=args).to(args.device)
        # optimizer = torch.optim.Adam(
        #     [{'params': model.gnn.parameters(), 'weight_decay': args.weight_decay, 'lr': args.gnn_lr},
        #      {'params': model.bert.parameters(), 'weight_decay': args.weight_decay, 'lr': args.bert_lr}, ])
        optimizer = build_optimizer(args, model)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=5, verbose=True)
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
        # logging.info(f"Number of parameters: trainable_params:{trainable_params}\n")
        wandb.config.optimizer = optimizer

        # -------------------------------------- train and test --------------------------------------
        if not os.path.exists(args.best_model_save_path):
            logging.info(f"args.best_model_save_path {args.best_model_save_path} Not Exists.")
        trainer = Trainer(model, optimizer, scheduler, args)
        trainer.fit(graph, evaluator, train_loader, bert_infer_loader, split_idx)

        res = trainer.test(graph, bert_infer_loader, evaluator, split_idx, mode="test")
        # for strategy, metrics in res['all_metrics'].items():
        #     logging.info(
        #         f" dataset: {args.data_name}, seed: {seed}, {strategy} | acc: {metrics['acc']:.4f}, f1: {metrics['f1']:.4f}, rocauc: {metrics['rocauc']:.4f}")
        # logging.info(
        #     f"dataset: {args.data_name}, seed: {seed}, Best Strategy: {res['best_strategy']}, Acc: {res['best_acc']:.4f}")
        acc = res['best_acc']
        if isinstance(acc, torch.Tensor):
            acc = acc.cpu().item()
        test_accs.append(acc)
        cleanup()

    test_acc_mean = np.mean(test_accs, axis=0)
    std = np.std(test_accs)
    values = np.asarray(test_accs, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    # logging.info(f'args: {args}')
    logging.info(
        f'dataset: {args.data_name}, test acc mean ± std = {test_acc_mean:.4f} ± {std:.4f}; test acc mean ± uncertainty = {test_acc_mean:.4f} ± {uncertainty:.4f}')


if __name__ == "__main__":
    # python run.py --data_name texas --gnn_name gcn --freeze_layers_count 3
    # python run.py --data_name cora --gnn_name gcn --freeze_layers_count 3
    #  python run.py --data_name cornell --gnn_name gcn --freeze_layers_count 3 --weight_loss_graph 0.1 --weight_loss_text 1.0 --fuse_way concat
    torch.rand(1).cuda()
    # 'sinkhorn','wasserstein', 'gromov_wasserstein', 'fused_gromov_wasserstein', 'entropic_gromov_wasserstein', 'coot'
    for ot_type in ['full']: # "", 'full',
        logging.info(f"Time:{datetime.now()}, start---------------------------------------------------------")
        args.hardk=True
        args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
        return_dict = {"result": float("-inf")}
        trial_params = {}
        args.model_name = f"{args.data_name}_pe{args.use_pe}_{args.bert_name}_{args.gnn_name}_multiot"
        logging.info(f"args: {args}")
        # print(f"args: {args}")
        args.ot_impl = ot_type
        try:
            main(args)
        except Exception as e:
            print(f"Error: {e}")
            continue
        cleanup()
# run_multiot.py --data_name cornell --gnn_name gcn --freeze_layers_count 3 --weight_loss_graph 0.01 --weight_loss_text 1.0 --fuse_way add
