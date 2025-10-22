import os
import torch
import argparse
import pickle
from tqdm.auto import tqdm
from pathlib import Path

from utils import set_seed, print_args, set_config_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel, LightGCN
from explainer import PaGELink

class DataMapper:
    def __init__(self, pt_file_path):
        self.original_data = torch.load(pt_file_path)
        self.num_users = len(self.original_data.user_id_to_node)

    def get_user_raw_text(self, dgl_user_id):
        return self.original_data.raw_texts[dgl_user_id]

    def get_item_title(self, dgl_item_id):
        return self.original_data.item_titles[dgl_item_id]
    
    def get_item_raw_text(self, dgl_item_id):
        return self.original_data.raw_texts[dgl_item_id + self.num_users]

parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=0)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--output_dir', type=str, default='saved_models')
parser.add_argument('--dataset_name', type=str, default='aug_citation')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')
parser.add_argument('--stage', type=str, default='step2')
parser.add_argument('--split', type=str, default='train')

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--saved_model_name', type=str, default='')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='user', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='item', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='likes', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')

'''
Explanation args
'''
parser.add_argument('--lr', type=float, default=0.01, help='explainer learning_rate') 
parser.add_argument('--alpha', type=float, default=1.0, help='explainer on-path edge regularizer weight') 
parser.add_argument('--beta', type=float, default=1.0, help='explainer off-path edge regularizer weight') 
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops') 
parser.add_argument('--num_epochs', type=int, default=10, help='How many epochs to learn the mask')
parser.add_argument('--num_paths', type=int, default=10, help='How many paths to generate')
parser.add_argument('--max_path_length', type=int, default=5, help='max lenght of generated paths')
parser.add_argument('--k_core', type=int, default=2, help='k for the k-core graph') 
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1') 
parser.add_argument('--save_explanation', default=True, action=argparse.BooleanOptionalAction, 
                    help='Whether to save the explanation')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'pagelink')    

args.src_ntype = 'user'
args.tgt_ntype = 'item'

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}
    
if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'
    
print_args(args)
set_seed(0)

processed_g = load_dataset(args.output_dir, args.split, args.valid_ratio, args.test_ratio, args.stage)[1]
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in processed_g]

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
#encoder = LightGCN(mp_g, args.emb_dim, 2)
#model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
#state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}_{args.split}_lightgcn.pth', map_location='cpu')
path = os.path.join(args.output_dir, f"gnn_{args.split}.pth")
state = torch.load(path, map_location='cpu')
model.load_state_dict(state)  

pagelink = PaGELink(model, 
                    lr=args.lr,
                    alpha=args.alpha, 
                    beta=args.beta, 
                    num_epochs=args.num_epochs,
                    log=True).to(device)


test_src_nids, test_tgt_nids = test_pos_g.edges()
test_ids = range(test_src_nids.shape[0])
if args.max_num_samples > 0:
    test_ids = test_ids[:args.max_num_samples]

# mapper = DataMapper('/home/yuhanli/lyh/GRec/data/yelp/data_trn.pt')

pred_edge_to_comp_g_edge_mask = {}
pred_edge_to_paths = {}
for i in tqdm(test_ids):
    src_nid, tgt_nid = test_src_nids[i].unsqueeze(0), test_tgt_nids[i].unsqueeze(0)
    
    with torch.no_grad():
        pred = model(src_nid, tgt_nid, mp_g).sigmoid().item() > 0.5

    if pred:
        src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
        try:
            paths, comp_g_edge_mask_dict = pagelink.explain(src_nid, 
                                                            tgt_nid, 
                                                            mp_g,
                                                            args.num_hops,
                                                            args.prune_max_degree,
                                                            args.k_core, 
                                                            args.num_paths, 
                                                            args.max_path_length,
                                                            return_mask=True)
        except Exception as e:
            print(f"Error explaining edge ({src_nid}, {tgt_nid}): {str(e)}")
            paths, comp_g_edge_mask_dict = [], {}
        
        pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict 
        pred_edge_to_paths[src_tgt] = paths

if args.save_explanation:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    saved_edge_explanation_file = f'pagelink_{args.saved_model_name}_{args.split}_pred_edge_to_comp_g_edge_mask'
    saved_path_explanation_file = f'pagelink_{args.saved_model_name}_{args.split}_pred_edge_to_paths'
    pred_edge_to_comp_g_edge_mask = {edge: {k: v.cpu() for k, v in mask.items()} for edge, mask in pred_edge_to_comp_g_edge_mask.items()}

    saved_edge_explanation_path = Path.cwd().joinpath(args.output_dir, saved_edge_explanation_file)
    with open(saved_edge_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_comp_g_edge_mask, f)

    saved_path_explanation_path = Path.cwd().joinpath(args.output_dir, saved_path_explanation_file)
    with open(saved_path_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_paths, f)


# python pagelink.py --dataset_name yelp --save_explanation --device_id 0