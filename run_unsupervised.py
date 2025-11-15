import logging
from datetime import datetime

import numpy as np
import seaborn as sns
import torch
from eval_tools import LRE
from utils.plots import plot_tsne
from wandb_wrapper import setup_wandb

from data.load_data import get_main_dataloader, data_attributes
from models.bert import load_pretrained_model_and_tokenizer_unspervised
from trainer_unsupervised import Trainer_unsupervised
from utils.arguments import args
from utils.path_config import initialize_paths_all
from utils.util import init_random_state, use_best_hyperparams, get_linear_schedule_with_warmup, cleanup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
    filename='main_unsupervised_otclout.log', filemode='a')


# torch.autograd.set_detect_anomaly(True)


def main(args):
    test_accs = []
    # randseed = [random.randint(0, 9999) for _ in range(10)]
    logging.info(f"Time:{datetime.now()}, ---------------------------------------------------------")
    for seed in range(10):
        args.seed = seed
        logging.info(f"args: {args}")
        init_random_state(seed=seed)
        initialize_paths_all(args, seed, fuse_way="add", unsupervised=True)
        wandb = setup_wandb(args, seed,
            suffix=f"{args.data_name}_{args.gnn_name}_{args.bert_name}_unsupervised_pe{args.use_pe}")

        # Step 1: Load data =================================================================== #
        # logging.info(f"Time:{datetime.now()}, ------------------------------- load data--------------------------")
        data_attr = data_attributes(args.data_name)
        args.num_nodes = data_attr['num_nodes']
        args.input_dim = data_attr['feat_dim']
        args.num_classes = data_attr['num_classes']
        _, tokenizer = load_pretrained_model_and_tokenizer_unspervised(args.current_dir, args.bert_name)
        train_loader, bert_infer_loader, gnn_infer_loader, graph, split_idx = get_main_dataloader(args, tokenizer,
            args.device, seed, use_text=True, use_pe=args.use_pe)

        # Step 2: Create model =================================================================== #
        model = OTCLUnsupervised(args=args).to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: trainable_params:{trainable_params}\n")

        # Step 3: Create training components ===================================================== #
        optimizer = torch.optim.Adam(
            [{'params': model.gnn.parameters(), 'weight_decay': args.weight_decay, 'lr': args.gnn_lr},
             {'params': model.bert.parameters(), 'weight_decay': args.weight_decay, 'lr': args.bert_lr}, ])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=5, verbose=True)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps),
            num_training_steps=total_steps)

        # Step 4: Training epochs ================================================================ #
        torch.cuda.empty_cache()
        trainer = Trainer_unsupervised(model, optimizer, scheduler, args)
        # if not os.path.exists(args.best_model_save_path):
        #     logging.info(f"Not Exists.")
        trainer.fit(graph, bert_infer_loader)
        # 移除钩子（如果不再需要捕获邻居信息）
        # trainer.hook_handle.remove()
        # Step 5:  Linear evaluation ========================================================== #
        model.load_state_dict(torch.load(args.best_model_save_path))
        model.eval()
        embeds = model.get_embedding(graph, bert_infer_loader)
        plot_tsne(embeds, graph.y, args.best_emb_tsne_path, title=None, seed=args.seed, use_pca=False, emb_type="umap")

        result = LRE(embeds, graph.y, graph.train_id, graph.val_id, graph.test_id)

        logging.info(
            f"dataset: {args.data_name}, seed: {seed}, train acc: {result['train_acc']:.4f}, val acc: {result['val_acc']:.4f}, test acc: {result['test_acc']:.4f}")
        test_accs.append(result['test_acc'])
        cleanup()

    test_acc_mean = np.mean(test_accs, axis=0)
    std = np.std(test_accs)
    values = np.asarray(test_accs, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

    logging.info(f'args: {args}')
    logging.info(
        f'dataset: {args.data_name}, test acc mean ± std = {test_acc_mean:.4f} ± {std:.4f}; test acc mean ± uncertainty = {test_acc_mean:.4f} ± {uncertainty:.4f}')


if __name__ == "__main__":
    # python run_unsupervised.py --data_name texas --gnn_name gcn --use_pe False --freeze_layers_count 0
    # python run_unsupervised.py --data_name cornell --gnn_name gcn --use_pe False --freeze_layers_count 0
    # python run_unsupervised.py --data_name Wisconsin --gnn_name gcn --use_pe False --freeze_layers_count 0
    # python run_unsupervised.py --data_name cora --gnn_name gcn --use_pe False --batch_size 512 --patience 4 --dropout 0 --freeze_layers_count 5
    # python run_unsupervised.py --data_name Amazon --gnn_name gcn --use_pe False  --batch_size 64 --patience 4 --dropout 0.3 --freeze_layers_count 1
    # python run_unsupervised.py --data_name pubmed --gnn_name gcn --use_pe False  --batch_size 64 --patience 4 --dropout 0.3 --freeze_layers_count 1
    # python run_unsupervised.py --data_name Actor --gnn_name gcn --use_pe False  --batch_size 512 --patience 4 --dropout 0.3 --freeze_layers_count 1

    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    return_dict = {"result": float("-inf")}
    trial_params = {}
    args.saveq = 0
    args.model_name = f"{args.data_name}_pe{args.use_pe}_{args.bert_name}_{args.gnn_name}_unsupervised"

    args.fuse_way = "add"

    main(args)
    cleanup()
