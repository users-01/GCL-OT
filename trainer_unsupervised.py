import copy
import gc
import logging
import os
import sys

import torch
from alive_progress import alive_bar
# from models.GNNNeighborExtractor import NeighborExtractor2, NeighborExtractor3, OptimizedNeighborExtractor
from utils.plots import plot_tsne
from utils.util import save_checkpoint_scheduler, time_logger, load_checkpoint_scheduler
from wandb_wrapper import wandb
class OptimizedNeighborExtractor:
    def __init__(self, max_num_neighbors):
        self.max_num_neighbors = max_num_neighbors  # 每个节点最多的邻居数目

    def get_neighbor_tensor(self, edge_index, graph_emb):
        num_nodes = graph_emb.size(0)
        max_num_neighbors = self.max_num_neighbors
        embedding_dim = graph_emb.size(1)

        # 创建邻居特征张量，初始化为节点本身的特征
        neighbor_features = torch.repeat_interleave(graph_emb.unsqueeze(1), max_num_neighbors, dim=1).to(
            graph_emb.device)
        neighbor_mask = torch.zeros(num_nodes, max_num_neighbors, dtype=torch.int).to(graph_emb.device)

        # 为每个节点的第一个位置填充自身特征
        neighbor_mask[:, 0] = 1  # 标记节点自身为有效邻居

        # 获取边的源节点和目标节点
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        # 使用矢量化操作批量填充邻居特征
        for src_node, dst_node in zip(src_nodes, dst_nodes):
            self._fill_neighbors(src_node, dst_node, graph_emb, neighbor_features, neighbor_mask, num_nodes)

        return neighbor_features, neighbor_mask

    def _fill_neighbors(self, src_node, dst_node, graph_emb, neighbor_features, neighbor_mask, num_nodes):

        # Ensure that the node index is valid (within the range of num_nodes)
        if src_node >= num_nodes or dst_node >= num_nodes:
            return  # Skip invalid nodes

        # 为源节点和目标节点填充邻居特征
        for node, neighbor in [(src_node, dst_node), (dst_node, src_node)]:
            if node >= num_nodes:  # Ensure that node index does not exceed the number of nodes
                continue  # Skip invalid nodes
            for j in range(1, self.max_num_neighbors):
                if neighbor_mask[node, j] == 0:  # 确保索引不会超出范围
                    # if neighbor_mask[node, j] == 0:  # 找到空闲位置

                    neighbor_features[node, j] = graph_emb[neighbor]

                    neighbor_mask[node, j] = 1  # 标记该位置为有效
                    break

class Trainer_unsupervised:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.accumulation_steps = args.accumulation_steps
        self.device = args.device
        self.best_model_state = None  # 用于保存最好模型的权重
        # self.neighbor_extractor = NeighborExtractor()
        # self.hook_handle = self.model.gnn.conv1.register_message_forward_hook(self.neighbor_extractor.message_hook)

        self.neighbor_extractor = OptimizedNeighborExtractor(max_num_neighbors=args.max_neighbors)

    def train_one_epoch(self,  data_loader):
        self.model.train()
        epoch_total_loss = 0.0
        accumulation_steps = self.args.accumulation_steps  # Number of batches to accumulate gradients over
        self.optimizer.zero_grad()

        loss = torch.tensor(0.0)
        for batch_idx, batch in enumerate(data_loader):
            # print(batch)
            for s in ["x", "y", "edge_index","input_ids","attention_mask"]:
                batch[s] = batch[s].to(self.device)
            batch_size = batch['batch_size']

            # Forward pass
            text_emb, graph_emb, fuse_emb, token_emb = self.model(
                batch_size=batch_size, x=batch["x"], edge_index=batch["edge_index"],
                text_input=batch["input_ids"][:batch_size], attention_mask=batch["attention_mask"][:batch_size]
                )
            # batch_neighbors_features：[batch_num_nodes, max_neighbors, num_node_features]
            # batch_neighbors_mask：[batch_num_nodes, max_neighbors]
            batch_neighbors_features, batch_neighbors_mask = self.neighbor_extractor.get_neighbor_tensor(
                batch["edge_index"], graph_emb)
            # print(batch_neighbors_features.shape, batch_neighbors_mask.shape)
            # print("邻居特征 shape:", batch_neighbors_features.shape)
            # print("掩码 shape:", batch_neighbors_mask.shape)
            # exit(0)
            loss = self.model.get_unsupervised_loss(
                text_emb=text_emb,
                token_emb=token_emb,
                graph_emb=graph_emb,
                batch_neighbors_features=batch_neighbors_features, # []
                batch_neighbors_mask=batch_neighbors_mask,
                attention_mask=batch["attention_mask"][:batch_size])

            # conv = self.model.extractor.get_first_message_layer()
            # batch_neighbors_features, batch_neighbors_mask = self.model.extractor.get_neighbor_tensor2(conv,
            #     edge_index=batch["edge_index"])
            # 打印结果
            # print("邻居特征 shape:", batch_neighbors_features.shape)
            # print("掩码 shape:", batch_neighbors_mask.shape)
            # print(f"batch['edge_index'].size():{batch['edge_index'].size()}, batch['x'].size():{batch['x'].size()}")
            #
            # print(f"捕获的邻居特征:", len(self.neighbor_extractor.neighbor_features), self.neighbor_extractor.neighbor_features[0].shape)
            # for i in range(len(self.neighbor_extractor.neighbor_features)):
            #     print(f"捕获的邻居特征:", self.neighbor_extractor.neighbor_features[i].shape)
            # loss = self.model.get_unsupervised_loss(
            #     text_emb,
            #     graph_emb,
            #     batch.precomputed_neighbor_features[:batch_size].to(self.device),
            #     token_emb,
            #     batch.graph_mask_outputs[:batch_size].to(self.device),
            #     batch["attention_mask"][:batch_size]
            #     )
            loss = loss / accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(data_loader)):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Accumulate loss for logging
            epoch_total_loss += loss.item()
        num_batches = len(data_loader)

        return epoch_total_loss / num_batches

    @time_logger
    def fit(self, graph, data_loader):
        start_epoch = 0
        patience_counter = 0
        best_loss = sys.float_info.max
        self.args.saveq=0

        # continue train
        if os.path.exists(self.args.checkpoint_path):
            logging.info(f"Loading checkpoint from {self.args.checkpoint_path}")
            start_epoch, _, best_valid_loss = load_checkpoint_scheduler(self.model, self.optimizer, None,
                self.args.checkpoint_path, scheduler=self.scheduler)
            self.best_model_state = copy.deepcopy(self.model.state_dict())

        with alive_bar(self.args.epochs) as bar:
            for epoch in range(start_epoch, self.args.epochs):
                train_loss = self.train_one_epoch(data_loader)  # logs
                # logging.info(
                #     f"Data: {self.args.data_name}, Seed: {self.args.seed}, Epoch: {epoch}, Train loss: {train_loss:.4f}")
                wandb.log({"Epoch": epoch, "train_loss": train_loss})
                if best_loss > train_loss:
                    patience_counter = 0
                    best_loss = train_loss
                    # logging.info(f"Seed: {self.args.seed}, Epoch: {epoch}, train_loss: {train_loss:.4f}")

                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    save_checkpoint_scheduler(self.model, self.optimizer, epoch, best_loss, self.args.checkpoint_path,
                        scheduler=self.scheduler)
                else:
                    patience_counter += 1
                if patience_counter >= self.args.patience:
                    # logging.info(f"Early stopping at epoch {epoch}. Best best_loss={best_loss:.4f}")
                    break

                bar()
                # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                # gc.collect()
                # torch.cuda.empty_cache()

            torch.save(self.best_model_state, str(self.args.best_model_save_path))
