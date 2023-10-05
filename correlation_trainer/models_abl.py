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
    def __init__(self, in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale):
        super(EnsembleGATDGFLayer, self).__init__()
        
        # Instantiate both modules
        self.dense_graph_flow = DenseGraphFlow(in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale)
        self.graph_attention_layer = GraphAttentionLayer(in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale)

    def forward(self, inputs, adj, op_emb):
        # Get outputs from both modules
        dense_output = self.dense_graph_flow(inputs, adj, op_emb)
        gat_output = self.graph_attention_layer(inputs, adj, op_emb)
        
        # Average the outputs
        ensemble_output = (dense_output + gat_output) / 2
        
        return ensemble_output

# DenseGraphFlow = SGConv
class DenseGraphFlow(nn.Module):

    def __init__(self, in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale):
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
    def __init__(self, in_features, out_features, op_emb_dim, residual, unique_attn_proj, opattention, leakrelu, attention_rescale):
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
            dual_input = False,
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
            updateopemb_scale = 0.1,
            nn_emb_dims = 128,
            input_zcp = False,
            zcp_embedder_dims = [128, 128],
            gtype = 'dense',
            residual = True,
            unique_attention_projection = False,
            opattention = True,
            leakyrelu = True,
            attention_rescale = False,
    ):
        super(GIN_Model, self).__init__()
        # if num_time_steps > 1:
        #     raise NotImplementedError
        self.device = device
        self.dual_input = dual_input
        self.wd_repr_dims = 8
        self.dinp = 2
        self.dual_gcn = dual_gcn
        self.num_zcps = num_zcps
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.zcp_embedding_dim = zcp_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.num_time_steps = num_time_steps
        self.fb_conversion_dims = fb_conversion_dims
        self.back_opemb = back_opemb
        self.back_y_info = back_y_info
        self.backward_gcn_out_dims = backward_gcn_out_dims
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
        reg_inp_dims = self.nn_emb_dims
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
        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        # gcn
        self.gcns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                LayerType(
                    in_dim, dim, self.op_embedding_dim, self.residual, self.unique_attn_proj, self.opattention, self.leakyrelu, self.attention_rescale
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

        # backward gcn
        self.b_gcns = []
        in_dim = self.fb_conversion_dims[-1]
        for dim in self.backward_gcn_out_dims:
            self.b_gcns.append(
                BackLayerType(
                    in_dim, dim, self.op_embedding_dim, self.residual, self.unique_attn_proj, self.opattention, self.leakyrelu, self.attention_rescale
                )
            )
            in_dim = dim
        self.b_gcns = nn.ModuleList(self.b_gcns)
        self.num_b_gcn_layers = len(self.b_gcns)

        # replace_bgcn_mlp_dims
        if self.back_mlp: # Generate a simple FullyConnectedNN with the replace_bgcn_mlp_dims
            self.replace_bgcn_mlp = FullyConnectedNN([self.fb_conversion_dims[-1]] + self.replace_bgcn_mlp_dims)

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
        if self.back_opemb:
            in_dim += self.op_embedding_dim
        if self.back_y_info:
            in_dim += self.gcn_out_dims[-1]
        if self.back_mlp:
            in_dim += self.replace_bgcn_mlp_dims[-1]
        else:
            in_dim += self.backward_gcn_out_dims[-1]
        # if self.back_mlp:
        #     in_dim = self.gcn_out_dims[-1] + self.replace_bgcn_mlp_dims[-1] + self.op_embedding_dim 
        # else:
        #     in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] + self.op_embedding_dim
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

        
    def embed_and_transform_arch(self, archs):
        # If self.dual input, remove first 2 and use input_op_emb.
        adjs = self.input_op_emb.new([arch.T for arch in archs[0]])
        # import pdb; pdb.set_trace()
        op_inds = self.input_op_emb.new([arch for arch in archs[1]]).long()
        op_embs = self.op_emb(op_inds)
        # Remove the first and last index of op_emb 
        # shape is [128, 7, 48], remove [128, 0, 48] and [128, 6, 48]
        if self.dual_input:
            op_embs = op_embs[:, self.dinp:-1, :]
            op_inds = op_inds[:, self.dinp:-1]
        else:
            op_embs = op_embs[:, 1:-1, :]
            op_inds = op_inds[:, 1:-1]
        b_size = op_embs.shape[0]
        if self.dual_input:
        # if False:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1])
                ), dim = 1
            )
            node_embs = torch.cat(
                (
                    self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 2, 1])
                ), dim = 1
            )
        else:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1])
                ), dim = 1
            )
            node_embs = torch.cat(
                (
                    self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1])
                ), dim = 1
            )

        x = self.x_hidden(node_embs)
        return adjs, x, op_embs, op_inds

    def _forward_pass(self, x, adjs, auged_op_emb):
        # --- forward pass ---
        y = x
        # adjs.nonzero().t().contiguous()
        for i_layer, gcn in enumerate(self.gcns):
            # import pdb; pdb.set_trace()
            y = gcn(y, adjs, auged_op_emb)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training = self.training)
        return y

    def _backward_pass(self, y, adjs, auged_op_emb):
    # If activating, define b_gcns, fb_conversion
        # --- backward pass ---
        if self.back_mlp:
            b_info = y[:, -1:, :].squeeze()
            b_y = self.replace_bgcn_mlp(b_info)
        else:
            b_info = y[:, -1:, :]
            # start backward flow
            b_info = self.fb_conversion(b_info)
            b_info = torch.cat(
                (
                    torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device = y.device),
                    b_info
                ),
                dim = 1
            )
            b_adjs = adjs.transpose(1, 2)
            b_y = b_info
            for i_layer, gcn in enumerate(self.b_gcns):
                b_y = gcn(b_y, b_adjs, auged_op_emb)
                if i_layer != self.num_b_gcn_layers - 1:
                    b_y = F.relu(b_y)
                    b_y = F.dropout(b_y, self.dropout, training = self.training)
        return b_y

    def _update_op_emb(self, y, b_y, op_emb):
    # If activating, define updateop_embedder
        # --- UpdateOpEmb ---
        if self.back_mlp:
            if self.back_opemb and self.back_y_info:
                in_embedding = torch.cat(
                    (
                        op_emb.detach(),
                        y.detach(),
                        b_y.unsqueeze(dim=1).repeat(1, y.shape[1], 1)
                    ),
                    dim = -1)
            elif self.back_opemb==False and self.back_y_info:
                in_embedding = torch.cat(
                    (
                        y.detach(),
                        b_y.unsqueeze(dim=1).repeat(1, y.shape[1], 1)
                    ),
                    dim = -1)
            elif self.back_opemb and self.back_y_info==False:
                in_embedding = torch.cat(
                    (
                        op_emb.detach(),
                        b_y.unsqueeze(dim=1).repeat(1, y.shape[1], 1)
                    ),
                    dim = -1)
            else:
                in_embedding = b_y.unsqueeze(dim=1).repeat(1, y.shape[1], 1)
        else:
            if self.back_opemb and self.back_y_info:
                in_embedding = torch.cat(
                    (
                        op_emb.detach(),
                        y.detach(),
                        b_y
                    ),
                    dim = -1)
            elif self.back_opemb==False and self.back_y_info:
                in_embedding = torch.cat(
                    (
                        y.detach(),
                        b_y
                    ),
                    dim = -1)
            elif self.back_opemb and self.back_y_info==False:
                in_embedding = torch.cat(
                    (
                        op_emb.detach(),
                        b_y
                    ),
                    dim = -1)
            else:
                in_embedding = b_y
        update = self.updateop_embedder(in_embedding)
        op_emb = op_emb + self.updateopemb_scale * update
        return op_emb

    def _final_process(self, y, op_inds):
        if self.dual_input:
            y = y[:, self.dinp:, :]
        else:
            y = y[:, 1:, :]
        y = torch.cat(
            (
                y[:, :-1, :] * (
                    op_inds != self.none_op_ind
                    )[:, :, None].to(torch.float32),
                y[:, -1:, :],
            ),
            dim = 1
        )
        y = torch.mean(y, dim = 1)
        return y

    def forward(self, x_ops_1=None, x_adj_1=None, x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=None):
        archs_1 = [[np.asarray(x.cpu()) for x in x_adj_1], [np.asarray(x.cpu()) for x in x_ops_1]]
        if zcp is not None:
            zcp = zcp.to(self.device)
        adjs_1, x_1, op_emb_1, op_inds_1 = self.embed_and_transform_arch(archs_1)
        adjs_1, x_1, op_emb_1, op_inds_1 = adjs_1.to(self.device), x_1.to(self.device), op_emb_1.to(self.device), op_inds_1.to(self.device)
        for tst in range(self.num_time_steps):
            y_1 = self._forward_pass(x_1, adjs_1, op_emb_1)
            if tst == self.num_time_steps - 1:
                break
            b_y_1 = self._backward_pass(y_1, adjs_1, op_emb_1)
            # import pdb; pdb.set_trace()
            op_emb_1 = self._update_op_emb(y_1, b_y_1, op_emb_1)
        y_1 = self._final_process(y_1, op_inds_1)
        if self.dual_gcn:
            archs_2 = [[np.asarray(x.cpu()) for x in x_adj_2], [np.asarray(x.cpu()) for x in x_ops_2]]
            adjs_2, x_2, op_emb_2, op_inds_2 = self.embed_and_transform_arch(archs_2)
            adjs_2, x_2, op_emb_2, op_inds_2 = adjs_2.to(self.device), x_2.to(self.device), op_emb_2.to(self.device), op_inds_2.to(self.device)
            for tst in range(self.num_time_steps):
                y_2 = self._forward_pass(x_2, adjs_2, op_emb_2)
                if tst == self.num_time_steps - 1:
                    break
                b_y_2 = self._backward_pass(y_2, adjs_2, op_emb_2)
                op_emb_2 = self._update_op_emb(y_2, b_y_2, op_emb_2)
            y_2 = self._final_process(y_2, op_inds_2)
            # y_1 += y_2
            y_1 = self.y_combiner(torch.cat((y_1, y_2), dim = -1))
        y_1 = y_1.squeeze()
        if self.input_zcp:
            zcp = self.zcp_embedder(zcp)
            if len(zcp.shape) == 1:
                zcp = zcp.unsqueeze(0)
            if len(y_1.shape) == 1:
                y_1 = y_1.unsqueeze(0)
            y_1 = torch.cat((y_1, zcp), dim = -1)
        if self.dual_gcn:
            norm_w_d = norm_w_d.to(self.device)
            norm_w_d = self.norm_wd_embedder(norm_w_d)
            if len(y_1.shape) == 1:
                y_1 = y_1.unsqueeze(0)
            if len(norm_w_d.shape) == 1:
                norm_w_d = norm_w_d.unsqueeze(0)
            y_1 = torch.cat((y_1, norm_w_d), dim = -1)
        y_1 = self.mlp(y_1)
        return y_1