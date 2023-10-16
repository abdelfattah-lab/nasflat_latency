import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
# from EGNN.models.EGNN_layer import EGNNConv
# from EGNN.models.GCNII_layer import GCNIIdenseConv
# from EGNN.models.SGC_layer import SGConv
# from EGNN.models.APPNP import APPNP

class FullyConnectedNN(nn.Module):
    def __init__(self, layer_sizes):
        super(FullyConnectedNN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))  # Add batch normalization
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NodeFeatureNN(nn.Module):
    def __init__(self, layer_sizes):
        super(NodeFeatureNN, self).__init__()
        self.fully_connected = FullyConnectedNN(layer_sizes)

    def forward(self, x):
        # Reshape the tensor so each node's features are a separate row
        batch_size, nodes, node_features = x.shape
        x_reshaped = x.view(batch_size * nodes, node_features)

        # Apply the FullyConnectedNN
        out = self.fully_connected(x_reshaped)

        # Reshape the tensor back to the original shape
        _, new_features = out.shape
        out_reshaped = out.view(batch_size, nodes, new_features)

        return out_reshaped

class EnsembleGATDGFLayer(nn.Module):
    def __init__(self, in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale, ensemble_fuse_method):
        super(EnsembleGATDGFLayer, self).__init__()
        self.ensemble_fuse_method = ensemble_fuse_method
        ensemble_conversion_dims = [64, 64]
        self.ensemble_conversion_dims = ensemble_conversion_dims
        # Instantiate both modules
        self.dense_graph_flow = DenseGraphFlow(in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale)
        self.graph_attention_layer = GraphAttentionLayer(in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale)
        
        if ensemble_fuse_method == "mlp":
            self.ensemble_conversion_list = []
            dim = self.gcn_out_dims[-1]
            num_fb_layers = len(self.ensemble_conversion_dims)
            for i_dim, ensemble_conversion_dim in enumerate(ensemble_conversion_dims):
                self.ensemble_conversion_list.append(nn.Linear(dim, ensemble_conversion_dims))
                if i_dim < num_fb_layers - 1:
                    self.ensemble_conversion_list.append(nn.ReLU(inplace=False))
                dim = ensemble_conversion_dim
            self.ensemble_conversion = nn.Sequential(*self.ensemble_conversion_list)

    def forward(self, inputs, adj, op_emb):
        # Get outputs from both modules
        dense_output = self.dense_graph_flow(inputs, adj, op_emb)
        gat_output = self.graph_attention_layer(inputs, adj, op_emb)
        
        if self.ensemble_fuse_method == 'add':
            # Average the outputs
            ensemble_output = (dense_output + gat_output) / 2   
        else:
            # Concatenate the outputs
            ensemble_output = torch.cat((dense_output, gat_output), dim=-1)
            # Apply a fully connected layer
            ensemble_output = self.ensemble_conversion(ensemble_output)
        return ensemble_output

# DenseGraphFlow = SGConv
class DenseGraphFlow(nn.Module):

    def __init__(self, in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale, ensemble_fuse_method):
        super(DenseGraphFlow, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.residual = residual
        self.unique_attn_proj = unique_attn_proj
        self.opattention = opattention
        self.leakrelu = leakrelu

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.op_attention = nn.Linear(op_emb_dim, out_features)
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, op_emb): # Why is inputs shape 8, when adj is 7?
        adj_aug = adj
        support = torch.matmul(inputs, self.weight) # no mismatch here, adj-shape propagated
        if self.opattention:
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support)
        else:
            output = torch.sigmoid(op_emb) * torch.matmul(adj_aug, support)
        if self.residual:
            output += support
        return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale, ensemble_fuse_method):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.unique_attn_proj = unique_attn_proj
        self.opattention = opattention
        self.leakrelu = leakrelu
        self.attention_rescale = attention_rescale
        self.op_attention = nn.Linear(op_emb_dim, out_features)
        self.Wk = nn.Linear(in_features, out_features, bias=False)
        if self.unique_attn_proj:
            self.Wv = nn.Linear(in_features, out_features, bias=False)
            self.Wq = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.layernorm = nn.LayerNorm(out_features)

    def forward(self, h, adj, op_emb):
        Whk = self.Wk(h) # model actual attention -> 3 different Ws, remove leakyrelu
        if self.unique_attn_proj:
            Whv = self.Wv(h)
            Whq = self.Wq(h)
        else:
            Whv = Whk
            Whq = Whk
        a_input = torch.einsum('balm,beam->belm', Whk.unsqueeze(-3).expand(-1, -1, Whk.size(1), -1), 
            Whq.unsqueeze(-2).expand(-1, Whq.size(1), -1, -1))
        if self.attention_rescale:
            a_input = a_input / np.sqrt(self.out_features)
        if self.leakrelu:
            alpha = F.leaky_relu(self.a(a_input))
        else:
            alpha = self.a(a_input)
        alpha = alpha * adj.unsqueeze(-1)
        attention = F.softmax(alpha, dim=-2)
        if self.opattention:
            h_prime = torch.sigmoid(self.op_attention(op_emb)) * torch.einsum('bijl,bjl->bil', attention, Whv)
        else:
            h_prime = torch.sigmoid(op_emb) * torch.einsum('bijl,bjl->bil', attention, Whv)
        # if self.residual:
        #     h_prime += Whk
        h_prime = self.layernorm(h_prime)
        return h_prime
    
class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, op_emb_dim):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        n_heads = 4
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(GraphAttentionLayer(in_features, out_features, op_emb_dim))
        self.n_heads = n_heads

    def forward(self, h, adj, op_emb):
        # Compute attention for each head
        outputs = [head(h, adj, op_emb) for head in self.heads]
        return torch.mean(torch.stack(outputs), dim=0)


class GIN_Model(nn.Module):
    def __init__(
            self,
            device='cpu',
            back_dense=False,
            dual_gcn = False,
            num_zcps = 13,
            vertices = 7,
            none_op_ind = 3,
            op_embedding_dim = 48,
            node_embedding_dim = 48,
            zcp_embedding_dim = 48,
            hid_dim = 96,
            gcn_out_dims = [128, 128, 128, 128, 128],
            mlp_dims = [200, 200, 200],
            dropout = 0.0,
            num_time_steps = 1,
            fb_conversion_dims = [128, 128],
            backward_gcn_out_dims = [128, 128, 128, 128, 128],
            replace_bgcn_mlp_dims = [128, 128, 128, 128, 128],
            updateopemb_dims = [128],
            back_mlp = False,
            back_opemb  = False,
            back_y_info = False,
            unroll_fgcn = False,
            updateopemb_scale = 0.1,
            nn_emb_dims = 128,
            input_zcp = False,
            zcp_embedder_dims = [128, 128],
            ensemble_fuse_method = "add",
            gtype = 'dense',
            residual = True,
            unique_attention_projection = False,
            opattention = True,
            leakyrelu = True,
            bmlp_ally = False,
            randopupdate = False,
            opemb_direct = False,
            detach_mode = 'default',
            attention_rescale = False,
            back_opemb_only = False,
    ):
        super(GIN_Model, self).__init__()
        # if num_time_steps > 1:
        #     raise NotImplementedError
        """
        Drawing from prior ablation studies, we hard-code several design aspects of the model such that they are no longer controled from main_abl
        1. We DO NOT use op_emb = op_emb + scale * update, we simply set op_emb = update
        2. We maintain the 'backward MLP', NOT 'backward GCN'
        3. We maintain the 'Op Update' MLP, and DO NOT detach anything
        4. Fix time-step to 2, to indicate an 'unrolled' computation. Less messy.
        """
        self.device = device
        # self.dual_input = dual_input
        self.wd_repr_dims = 8
        self.dinp = 2
        self.dual_gcn = dual_gcn
        self.dual_input = dual_gcn
        self.unroll_fgcn = unroll_fgcn
        self.num_zcps = num_zcps
        self.randopupdate = randopupdate
        self.op_embedding_dim = op_embedding_dim
        self.back_opemb_only = back_opemb_only
        self.bmlp_ally = bmlp_ally
        self.detach_mode = detach_mode
        self.node_embedding_dim = node_embedding_dim
        self.zcp_embedding_dim = zcp_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.num_time_steps = num_time_steps
        self.fb_conversion_dims = fb_conversion_dims
        self.back_opemb = back_opemb
        self.opemb_direct = opemb_direct
        self.back_y_info = back_y_info
        self.backward_gcn_out_dims = backward_gcn_out_dims
        self.ensemble_fuse_method = ensemble_fuse_method
        self.updateopemb_dims = updateopemb_dims
        self.updateopemb_scale = updateopemb_scale
        self.mlp_dims = mlp_dims
        self.nn_emb_dims = nn_emb_dims
        self.input_zcp = input_zcp
        self.zcp_embedder_dims = zcp_embedder_dims
        self.vertices = vertices
        self.none_op_ind = none_op_ind
        self.gtype = gtype
        self.back_dense = back_dense
        self.residual = residual
        self.unique_attn_proj = unique_attention_projection
        self.opattention = opattention
        self.leakyrelu = leakyrelu
        self.attention_rescale = attention_rescale
        self.replace_bgcn_mlp_dims = replace_bgcn_mlp_dims
        self.back_mlp = back_mlp
        if not self.opattention:
            self.op_embedding_dim = self.gcn_out_dims[0]
            self.node_embedding_dim = self.gcn_out_dims[0]
            self.zcp_embedding_dim = self.gcn_out_dims[0]
            op_embedding_dim = self.gcn_out_dims[0]
            node_embedding_dim = self.gcn_out_dims[0]
            zcp_embedding_dim = self.gcn_out_dims[0]
        self.mlp_dropout = 0.1
        self.training = True
        
        if self.gtype == 'dense':
            LayerType = DenseGraphFlow
            BackLayerType = DenseGraphFlow
        elif self.gtype == 'gat':
            LayerType = GraphAttentionLayer
            BackLayerType = GraphAttentionLayer
        elif self.gtype == 'gat_mh':
            LayerType = MultiHeadGraphAttentionLayer
            BackLayerType = MultiHeadGraphAttentionLayer
        elif self.gtype == 'ensemble':
            LayerType = EnsembleGATDGFLayer
            BackLayerType = EnsembleGATDGFLayer
        else:
            raise NotImplementedError

        if self.back_dense:
            BackLayerType = DenseGraphFlow

        # regression MLP
        self.mlp = []
        reg_inp_dims = self.gcn_out_dims[-1]
        if self.input_zcp:
            reg_inp_dims += self.zcp_embedding_dim
        if self.dual_gcn:
            reg_inp_dims += self.wd_repr_dims
        dim = reg_inp_dims
        for hidden_size in self.mlp_dims:
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=self.mlp_dropout)))
            dim = hidden_size
        self.mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*self.mlp)
        
        # op embeddings
        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad = True
        )
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad = False
        )
        self.op_emb = nn.Embedding(128, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.unroll_fgcn:
            self.x_hidden = nn.Linear(self.node_embedding_dim, self.gcn_out_dims[0])
        else:
            self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)
        # gcn
        self.gcns = []
        if self.unroll_fgcn:
            in_dim = self.gcn_out_dims[0]
        else:
            in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                LayerType(
                    in_dim, dim, self.op_embedding_dim, self.residual, self.unique_attn_proj, self.opattention, self.leakyrelu, self.attention_rescale, self.ensemble_fuse_method
                      # potential issue
                )
            )
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

        # zcp
        self.zcp_embedder = []
        zin_dim = self.num_zcps
        for zcp_emb_dim in self.zcp_embedder_dims:
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(zin_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=self.mlp_dropout)
                )
            )
            zin_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(zin_dim, self.zcp_embedding_dim))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)

        self.replace_bgcn_mlp = []
        in_dim = self.fb_conversion_dims[-1]
        num_layers = len(self.replace_bgcn_mlp_dims)
        for i_dim, mlp_dim in enumerate(self.replace_bgcn_mlp_dims):
            self.replace_bgcn_mlp.append(nn.Linear(in_dim, mlp_dim))
            if i_dim < num_layers - 1:
                self.replace_bgcn_mlp.append(nn.ReLU(inplace=False))
            in_dim = mlp_dim
        self.replace_bgcn_mlp = nn.Sequential(*self.replace_bgcn_mlp)

        # fb_conversion
        if self.num_time_steps > 1:
            self.fb_conversion_list = []
            dim = self.gcn_out_dims[-1]
            num_fb_layers = len(self.fb_conversion_dims)
            for i_dim, fb_conversion_dim in enumerate(fb_conversion_dims):
                self.fb_conversion_list.append(nn.Linear(dim, fb_conversion_dim))
                if i_dim < num_fb_layers - 1:
                    self.fb_conversion_list.append(nn.ReLU(inplace=False))
                dim = fb_conversion_dim
            self.fb_conversion = nn.Sequential(*self.fb_conversion_list)
        
        # updateop_embedder
        self.updateop_embedder = []
        in_dim = 0
        in_dim += self.op_embedding_dim
        in_dim += self.replace_bgcn_mlp_dims[-1]
        if self.back_opemb_only:
            in_dim = self.op_embedding_dim
        for embedder_dim in self.updateopemb_dims:
            self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
            self.updateop_embedder.append(nn.ReLU(inplace = False))
            in_dim = embedder_dim
        self.updateop_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
        self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

        # combine y_1 and y_2
        if self.dual_gcn:
            self.y_combiner = nn.Linear(self.gcn_out_dims[-1] * 2, self.gcn_out_dims[-1])
            # add 1 relu and layer
            self.y_combiner = nn.Sequential(
                self.y_combiner,
                nn.ReLU(inplace = False),
                nn.Linear(self.gcn_out_dims[-1], self.gcn_out_dims[-1])
            )

        if self.dual_gcn:
            # wd embedder
            self.norm_wd_embedder = []
            in_dim = 2
            for embedder_dim in [self.wd_repr_dims]:
                self.norm_wd_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.norm_wd_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.norm_wd_embedder.append(nn.Linear(in_dim, self.wd_repr_dims))
            self.norm_wd_embedder = nn.Sequential(*self.norm_wd_embedder)

    def _concat_op_embs(self, b_size, op_embs, dual=False):
        base = [
            self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
            op_embs,
            self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1])
        ]
        if dual:
            base.insert(1, self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]))
        return torch.cat(base, dim=1)

    def _concat_node_embs(self, b_size, dual=False):
        vertices = self.vertices - (2 if dual else 1)
        base = [
            self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
            self.other_node_emb.unsqueeze(0).repeat([b_size, vertices, 1])
        ]
        if dual:
            base.insert(1, self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]))
        return torch.cat(base, dim=1)
    
    def _prepare_architecture(self, x_adj, x_ops):
        archs = [[np.asarray(x.cpu()) for x in x_adj], [np.asarray(x.cpu()) for x in x_ops]]
        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        return adjs.to(self.device), x.to(self.device), op_emb.to(self.device), op_inds.to(self.device)

    def _ensure_2d(self, tensor):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch.T for arch in archs[0]])
        op_inds = self.input_op_emb.new([arch for arch in archs[1]]).long()
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        
        if self.dual_input:
            op_embs = op_embs[:, self.dinp:-1, :]
            op_inds = op_inds[:, self.dinp:-1]
            op_embs = self._concat_op_embs(b_size, op_embs, dual=True)
            node_embs = self._concat_node_embs(b_size, dual=True)
        else:
            op_embs = op_embs[:, 1:-1, :]
            op_inds = op_inds[:, 1:-1]
            op_embs = self._concat_op_embs(b_size, op_embs)
            node_embs = self._concat_node_embs(b_size)
        
        x = self.x_hidden(node_embs)
        return adjs, x, op_embs, op_inds

    def _forward_pass(self, x, adjs, auged_op_emb):
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, auged_op_emb)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training = self.training)
        return y
    
    def _final_process(self, y, op_inds):
        if self.dual_input:
            y = y[:, self.dinp:, :]
        else:
            y = y[:, 1:, :]
        noneop_locs = (op_inds != self.none_op_ind)[:, :, None].to(torch.float32)
        processed_y = y[:, :-1, :] * noneop_locs
        y = torch.cat((processed_y, y[:, -1:, :],), dim = 1)
        y = torch.mean(y, dim = 1)
        return y

    def _process_architecture(self, x, adjs, op_emb, op_inds):
        y = self._forward_pass(x, adjs, op_emb)
        # Here, we use the last node output and the embedding itself, to re-represent the embedding.
        # and then do another forward pass with the new embedding.
        # this is essentially doing a double forward pass? 
        # Can we create a 'new' forward net, which is separate from the main forward net, just to process the opemb?
        if self.separate_fp:
            op_emb = self._forward_op_pass(x, adj, op_emb)
            y = self._forward_pass(x, adjs, op_emb)
            return self._final_process(y, op_inds)
        else:
            jlz = self.replace_bgcn_mlp(y[:, -1:, :].squeeze().unsqueeze(dim=1).repeat(1, y.shape[1], 1))
            in_embedding = torch.cat((op_emb, jlz), dim=-1)
            op_emb = self.updateop_embedder(in_embedding)
            y = self._forward_pass(x, adjs, op_emb)
            return self._final_process(y, op_inds)

    def _concat_embedded_input(self, main_tensor, input_tensor, embedder):
        input_tensor = embedder(input_tensor.to(self.device))
        input_tensor, main_tensor = self._ensure_2d(input_tensor), self._ensure_2d(main_tensor)
        return torch.cat((main_tensor, input_tensor), dim=-1)

    def forward(self, x_ops_1=None, x_adj_1=None, x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=None):
        adjs_1, x_1, op_emb_1, op_inds_1 = self._prepare_architecture(x_adj_1, x_ops_1)
        y_1 = self._process_architecture(x_1, adjs_1, op_emb_1, op_inds_1)
        
        if self.dual_gcn:
            adjs_2, x_2, op_emb_2, op_inds_2 = self._prepare_architecture(x_adj_2, x_ops_2)
            y_2 = self._process_architecture(x_2, adjs_2, op_emb_2, op_inds_2)
            y_1 = self.y_combiner(torch.cat((y_1, y_2), dim=-1))
        
        y_1 = y_1.squeeze()
        if self.input_zcp:
            y_1 = self._concat_embedded_input(y_1, zcp, self.zcp_embedder)
        if self.dual_gcn:
            y_1 = self._concat_embedded_input(y_1, norm_w_d, self.norm_wd_embedder)
        
        return self.mlp(y_1)
