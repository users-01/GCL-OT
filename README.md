# GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs

## Data: 
## Extension: 


## Quick start

```bash
# create a clean environment (CUDA >= 11.8 suggested)
conda create -n gclot python >=3.10 pytorch >=2.2 cudatoolkit >=11.8 -c pytorch -y
conda activate gclot
pip install -r requirements.txt        # generated separately

# example run on the Texas
python run.py \
  --data_name texas \
  --gnn_name gcn \
  --bert_name distilbert \
  --fuse_way add
````

All command-line flags are defined in `utils/arguments.py`, default hyper-parameters already reproduce the paper numbers.


```bash
python run.py --data_name texas
```
