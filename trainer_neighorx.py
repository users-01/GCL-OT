import copy
import logging
import os
import sys

import torch
from alive_progress import alive_bar

from utils.util import save_checkpoint_scheduler, time_logger, load_checkpoint_scheduler


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


class Trainer:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.patience = args.patience
        self.device = args.device
        self.best_model_state = None
        self.neighbor_extractor = OptimizedNeighborExtractor(max_num_neighbors=args.max_neighbors)

    @time_logger
    def train_one_epoch(self, graph, train_loader):
        self.model.train()
        epoch_total_loss = 0.0
        # total_loss = 0
        total_fuse_loss = 0
        total_text_loss = 0
        total_graph_loss = 0
        total_contra_loss = 0
        accumulation_steps = self.args.accumulation_steps  # Number of batches to accumulate gradients over
        self.optimizer.zero_grad()
        num_batches = len(train_loader)
        loss = torch.tensor(0.0)
        for batch_idx, batch in enumerate(train_loader):
            # print(f"batch_idx:{batch_idx}, batch:{batch}")
            for s in ["x", "y", "edge_index", "input_ids", "attention_mask","batch_neighbors_features","batch_neighbors_mask"]:
                batch[s] = batch[s].to(self.device)
            batch_size = batch['batch_size']

            fuse_logits, text_logits, graph_logits, logits_ensemb, text_emb, graph_emb, fuse_emb, fuse_emb2, token_emb = self.model(
                batch_size=batch_size, x=batch["x"], edge_index=batch["edge_index"],
                text_input=batch["input_ids"][:batch_size], attention_mask=batch["attention_mask"][:batch_size])

            loss, fuse_loss, text_loss, graph_loss, contra_loss = self.model.get_loss(fuse_logits, text_logits,
                graph_logits, batch["y"][:batch_size], text_emb, graph_emb, batch["batch_neighbors_features"][:batch_size], token_emb[:batch_size],
                batch["batch_neighbors_mask"][:batch_size], batch["attention_mask"][:batch_size])
            # out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            # y = batch.y[:batch.batch_size].squeeze()
            # loss = F.cross_entropy(out, y)
            # loss, fuse_loss, text_loss, graph_loss, contra_loss =  self.model.get_loss2(fuse_logits=fuse_logits, text_logits=text_logits, graph_logits=graph_logits, labels=batch.y[:batch_size], text_emb=text_emb, graph_emb=graph_emb, edge_index=batch.edge_index, token_emb=token_emb, attention_mask=batch["attention_mask"][:batch.batch_size], if_contrast_loss=True)

            loss = loss / accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Accumulate loss for logging
            epoch_total_loss += loss.item()
            total_fuse_loss += fuse_loss
            total_text_loss += text_loss
            total_graph_loss += graph_loss
            total_contra_loss += contra_loss
        # logs = {'total_loss': epoch_total_loss / num_batches, 'fuse_loss': total_fuse_loss / num_batches,
        #         'text_loss': total_text_loss / num_batches, 'graph_loss': total_graph_loss / num_batches,
        #         'contra_loss': total_contra_loss / num_batches}
        # return logs
        # print(f"epoch_total_loss")
        return epoch_total_loss / num_batches

    # @time_logger
    def fit(self, graph, evaluator, train_loader, bert_infer_loader, split_idx):
        start_epoch = 0
        patience_counter = 0
        best_valid_loss = sys.float_info.max
        self.args.saveq=0

        # 加载 checkpoint
        # if os.path.exists(self.args.checkpoint_path):
        #     logging.info(f"Loading checkpoint from {self.args.checkpoint_path}")
        #
        #     # start_epoch, _, best_valid_loss = load_checkpoint_2(self.model, self.optimizer, None, self.args.checkpoint_path)
        #     self.best_model_state = copy.deepcopy(self.model.state_dict())

        # 继续训练
        if os.path.exists(self.args.checkpoint_path):
            logging.info(f"Loading checkpoint from {self.args.checkpoint_path}, start_epoch:{start_epoch}, best_valid_loss:{best_valid_loss}")
            start_epoch, _, best_valid_loss = load_checkpoint_scheduler(self.model, self.optimizer, None,
                self.args.checkpoint_path, scheduler=self.scheduler)
            self.best_model_state = copy.deepcopy(self.model.state_dict())

        with alive_bar(self.args.epochs) as bar:
            for epoch in range(start_epoch, self.args.epochs):
                train_loss = self.train_one_epoch(graph, train_loader)  # logs
                if epoch % 5 == 0:
                    valid_loss = self.validate(graph, bert_infer_loader, evaluator, split_idx, mode="valid")
                    if valid_loss < best_valid_loss:
                        patience_counter = 0
                        best_valid_loss = valid_loss
                        save_checkpoint_scheduler(self.model, self.optimizer, epoch, valid_loss,
                            self.args.checkpoint_path, scheduler=self.scheduler)
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                    else:
                        patience_counter += 1
                    if patience_counter >= self.args.patience:
                        break
                    bar()
            torch.save(self.best_model_state, str(self.args.best_model_save_path))

    @torch.no_grad()
    def validate(self, graph, bert_infer_loader, evaluator, split_idx, mode="valid"):
        self.args.saveq = 0
        self.model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            node_id = graph.node_id.to(self.device)
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            y = graph.y.to(self.device)
            graph_emb, text_emb, fuse_emb, fuse_logits, graph_logits, text_logits, ensemb_logits = self.model.inference(
                self.device, node_id, x, edge_index, y, bert_infer_loader)
            fuse_pred = torch.argmax(fuse_logits, dim=1)

            split = split_idx[mode]
            valid_loss, fuse_loss, text_loss, graph_loss, contra_loss = self.model.get_loss(fuse_logits[split],
                text_logits[split], graph_logits[split], y[split], text_emb[split], graph_emb[split], None, None, None,
                None, if_contrast_loss=False)
            return valid_loss

    @torch.no_grad()
    def test(self, graph, bert_infer_loader, evaluator, split_idx, mode="test", ):
        self.args.saveq=1

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        # elif os.path.exists(self.args.best_model_save_path):
        #     # self.model.load_state_dict(self.args.best_model_save_path)
        #     try:
        #         self.model.load_state_dict(torch.load(self.args.best_model_save_path, map_location=y.device))
        #     except ValueError:
        #
        #         print("Warning: No best_model_state found. Using default model parameters.")
        else:
            print("Warning: No best_model_state found. Using current model parameters.")
        self.model.eval()
        with torch.no_grad():
            node_id = graph.node_id.to(self.device)
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            y = graph.y.to(self.device)
            graph_emb, text_emb, fuse_emb, fuse_emb2, fuse_logits, graph_logits, text_logits, ensemb_logits, precomputed_neighbor_features, token_emb_all, graph_mask_outputs, attention_mask_all = self.model.inference_test(
                self.device, node_id, x, edge_index, y, bert_infer_loader)
            graph_pred = torch.argmax(graph_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            fuse_pred = torch.argmax(fuse_logits, dim=1)
            ensemb_pred = torch.argmax(ensemb_logits, dim=1)

            split = split_idx[mode]
            graph_acc, graph_rocauc, graph_f1 = evaluator.eval(y, graph_pred, graph_logits, split, graph.num_classes)
            text_acc, text_rocauc, text_f1 = evaluator.eval(y, text_pred, text_logits, split, graph.num_classes)
            fuse_acc, fuse_rocauc, fuse_f1 = evaluator.eval(y, fuse_pred, fuse_logits, split, graph.num_classes)
            ensemb_acc, ensemb_rocauc, ensemb_f1 = evaluator.eval(y, ensemb_pred, ensemb_logits, split, graph.num_classes)
            metrics = {'graph': {'acc': graph_acc, 'rocauc': graph_rocauc, 'f1': graph_f1}, 'text': {'acc': text_acc, 'rocauc': text_rocauc, 'f1': text_f1}, 'fuse': {'acc': fuse_acc, 'rocauc': fuse_rocauc, 'f1': fuse_f1}, 'ensemb': {'acc': ensemb_acc, 'rocauc': ensemb_rocauc, 'f1': ensemb_f1}, }
            best_strategy = max(metrics.items(), key=lambda x: x[1]['acc'])[0]
            best_metrics = metrics[best_strategy]
            return {'all_metrics': metrics, 'best_acc': best_metrics['acc'], 'best_f1': best_metrics['f1'], 'best_rocauc': best_metrics['rocauc']}
