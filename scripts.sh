# Pre-training
nohup python -u pre_train.py --dataset_name Pubmed --task SimGRACE --gnn_type GCN > ./logs/pretrain_Pubmed_SimGRACE_GCN.log 2>&1 &
nohup python -u pre_train.py --dataset_name Cora --task SimGRACE --gnn_type GCN >> ./logs/pretrain_Cora_SimGRACE_GCN_dim256.log 2>&1 &
nohup python -u pre_train.py --dataset_name Citeseer --task SimGRACE --gnn_type GCN >> ./logs/pretrain_Citeseer_SimGRACE_GCN.log 2>&1 &



# Gprompt
## Cora
nohup python -u downstream_task_trojan.py --prompt_type Gprompt --compare_method Ours --dataset_name Cora --gnn_type GCN --pooling_type sum > ./logs/Gprompt_Ours_Cora_SimGRACE_GCN.log 2>&1 &
## Citeseer
nohup python -u downstream_task_trojan.py --prompt_type Gprompt --compare_method Ours --dataset_name Citeseer --gnn_type GCN --pooling_type sum --trianing_spilt k-shot --shot_num 100 > ./logs/Gprompt_Ours_Citeseer_SimGRACE_GCN.log 2>&1 &
## Pubmed
nohup python -u downstream_task_trojan.py --prompt_type Gprompt --compare_method Ours --dataset_name Pubmed --gnn_type GCN --pooling_type sum > ./logs/Gprompt_Ours_Pubmed_SimGRACE_GCN.log 2>&1 &

# GPPT
## Ours
nohup python -u downstream_task_trojan.py --prompt_type GPPT --compare_method Ours --dataset_name Cora --gnn_type GCN --pooling_type sum --trianing_spilt k-shot --shot_num 100 > ./logs/GPPT_Ours_Cora_SimGRACE_GCN_100_shot_meta_0911.log 2>&1 &
nohup python -u downstream_task_trojan.py --prompt_type GPPT --compare_method Ours --dataset_name Citeseer --gnn_type GCN --pooling_type sum --trianing_spilt k-shot --shot_num 100 > ./logs/GPPT_Ours_Citeseer_SimGRACE_GCN_100_shot.log 2>&1 &
nohup python -u downstream_task_trojan.py --prompt_type GPPT --compare_method Ours --dataset_name Pubmed --gnn_type GCN --pooling_type sum --trianing_spilt k-shot --shot_num 100 > ./logs/GPPT_Ours_Pubmed_SimGRACE_GCN_100_shot.log 2>&1 &


