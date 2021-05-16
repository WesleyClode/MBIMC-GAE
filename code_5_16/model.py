"""NN modules"""
import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import time
from utils import get_activation, to_etype_name
th.set_printoptions(profile="full")

class GCMCGraphConv(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        #self.att = nn.Parameter(th.Tensor(1 , basis_units))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        #init.xavier_uniform_(self.att)

    def forward(self, graph, feat, weight=None, Two_Stage = False):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat      # dst feature not used
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
            #weight = th.matmul(self.att, weight.view(weight.shape[0], -1)).view(weight.shape[1], -1)
            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)
            #feat = feat #* self.dropout(cj)
            '''
            if Two_Stage: 
                feat = th.cat([feat[:, :self._out_feats], feat[:, self._out_feats:self._out_feats*2], feat[:,self._out_feats*2:]], 1)
            else:
                feat = th.cat([feat[:, :self._out_feats].detach(), feat[:, self._out_feats:self._out_feats*2], feat[:,self._out_feats*2:]], 1)
            '''
            #feat = th.cat([feat[:, :self._out_feats] * self.dropout(cj), feat[:, self._out_feats:self._out_feats*2], feat[:,self._out_feats*2:]], 1)
            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            #rst = th.cat([rst[:, :self._out_feats] * ci, rst[:, self._out_feats:self._out_feats*2], rst[:,self._out_feats*2:]], 1)
            rst = rst * ci 
        #starttime = time.time()
        #graph.srcdata['reg'] = feat
        #graph.apply_edges(udf_u_mul_e_norm)
        #endtime = time.time()
        return rst

class GCMCLayer(nn.Module):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}

    The equation is applied to both user nodes and movie nodes and the parameters
    are not shared unless ``share_user_item_param`` is true.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    user_in_units : int
        Size of user input feature
    movie_in_units : int
        Size of movie input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output user and movie features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_user_item_param : bool, optional
        If true, user node and movie node share the same set of parameters.
        Require ``user_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """
    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None,
                 share_user_item_param=False,
                 ini = True,
                 basis_units=4,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        self.user_in_units = user_in_units
        self.msg_units = msg_units
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        # if ini:
        #     msg_units = msg_units  // 3
        if ini:
            msg_units = msg_units
        self.ini = ini
        self.msg_units = msg_units
        self.dropout = nn.Dropout(dropout_rate)
        #self.W_r = nn.ParameterDict()
        self.W_r = {}
        subConv = {}
        self.basis_units = basis_units
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units))
        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, msg_units))
        for i, rating in enumerate(rating_vals):
            # PyTorch parameter name can't contain "."
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                #self.W_r_union = nn.Parameter(th.randn(user_in_units, msg_units))
                #self.W_r[rating] = th.tensor(float(rating)) * W_r_union
                #self.W_r[rating] = self.W[i, :, :]
                #self.W_r['rev-%s' % rating] = self.W_r[rating]
                #self.W_r[rating] = nn.Parameter(th.randn(user_in_units, msg_units))
                #self.W_r['rev-%s' % rating] = self.W_r[rating]
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=False,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, graph, ufeat=None, ifeat=None, Two_Stage = False):
        """Forward function

        Parameters
        ----------
        graph : DGLHeteroGraph
            User-movie rating graph. It should contain two node types: "user"
            and "movie" and many edge types each for one rating value.
        ufeat : torch.Tensor, optional
            User features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            Movie features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : torch.Tensor
            New user features
        new_ifeat : torch.Tensor
            New movie features
        """
        in_feats = {'user' : ufeat, 'movie' : ifeat}
        mod_args = {}
        self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
        self.W = self.W.view(-1, self.user_in_units, self.msg_units)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            #mod_args[rating] = (self.basis,)
            #mod_args[rev_rating] = (self.basis,)
            mod_args[rating] = (self.W[i,:,:] if self.W_r is not None else None, Two_Stage)
            mod_args[rev_rating] = (self.W[i,:,:] if self.W_r is not None else None, Two_Stage)
            #mod_args[rating] = (self.W_r[rating] if self.W_r is not None else None, Two_Stage)
            #mod_args[rev_rating] = (self.W_r[rev_rating] if self.W_r is not None else None, Two_Stage)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        ufeat = out_feats['user']
        ifeat = out_feats['movie']
        if in_feats['user'].shape == ufeat.shape:
            ufeat = ufeat.view(ufeat.shape[0], -1) #+ 0.1 * in_feats['user']
            ifeat = ifeat.view(ifeat.shape[0], -1) #+ 0.1 * in_feats['movie']
        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat

def udf_u_mul_e_norm(edges):
    return {'reg' : edges.src['reg'] * edges.dst['ci']}
    #out_feats = edges.src['reg'].shape[1] // 3
    #return {'reg' : th.cat([edges.src['reg'][:, :out_feats] * edges.dst['ci'], edges.src['reg'][:, out_feats:out_feats*2], edges.src['reg'][:, out_feats*2:]], 1)}

def udf_u_mul_e(edges):
    return {'m' : th.cat([edges.src['h'], edges.dst['h']], 1)}

class MLPDecoder(nn.Module):
    r"""MLP decoder.

    Given a bipartite graph G, for each edge (i, j) ~ G, compute the likelihood
    of it being class r by:

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 num_basis=2,
                 dropout_rate=0.0):
        super(MLPDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(2 * in_units, 128)
        self.lin2 = nn.Linear(128, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self, graph, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        graph : DGLHeteroGraph
            "Flattened" user-movie graph with only one edge type.
        ufeat : th.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (|V_m|, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge.
        """
        with graph.local_scope():
            #ufeat = self.dropout(ufeat)
            #ifeat = self.dropout(ifeat)
            graph.nodes['movie'].data['h'] = ifeat
            graph.nodes['user'].data['h'] = ufeat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']
            #out = th.tanh(out)
            out = F.relu(self.lin1(out))
            #out = self.dropout(out)
            out = self.lin2(out)
        return out

class BiDecoder(nn.Module):
    r"""Bi-linear decoder.

    Given a bipartite graph G, for each edge (i, j) ~ G, compute the likelihood
    of it being class r by:

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 num_basis=2,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = nn.ParameterList(
            nn.Parameter(th.randn(in_units, in_units))
            for _ in range(num_basis))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        graph : DGLHeteroGraph
            "Flattened" user-movie graph with only one edge type.
        ufeat : th.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (|V_m|, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge.
        """
        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            #print("ifeat:",ifeat.shape)
            graph.nodes['movie'].data['h'] = ifeat
            basis_out = []
            for i in range(self._num_basis):
                graph.nodes['user'].data['h'] = ufeat @ self.Ps[i]
                graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
                basis_out.append(graph.edata['sr'])
            out = th.cat(basis_out, dim=1)
            out = self.combine_basis(out)
        return out

class DenseBiDecoder(nn.Module):
    r"""Dense bi-linear decoder.

    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for source
    nodes; one for destination nodes).

    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 num_basis=2,
                 dropout_rate=0.0):
        super().__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ufeat, ifeat):
        """Forward function.

        Compute logits for each pair ``(ufeat[i], ifeat[i])``.

        Parameters
        ----------
        ufeat : th.Tensor
            User embeddings. Shape: (B, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (B, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = th.einsum('ai,bij,aj->ab', ufeat, self.P, ifeat)
        out = self.combine_basis(out)
        return out

# def dot_or_identity(A, B, device=None):
#     # if A is None, treat as identity matrix
#     if A is None:
#         return B
#     elif A.shape[1] == 3:
#         if device is None:
#             return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
#         else:
#             #return th.cat([B[A[:, 0].long()], B[A[:, 1].long()]], 1).to(device)
#             return th.cat([B[A[:, 0].long()], B[A[:, 1].long()] , B[A[:, 2].long()]], 1).to(device)
#     else:
#         return A

def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif A.shape[1] == 1:
        if device is None:
            return th.cat([B[A[:, 0].long()]], 1)
        else:
            #return th.cat([B[A[:, 0].long()], B[A[:, 1].long()]], 1).to(device)
            return th.cat([B[A[:, 0].long()]], 1).to(device)
    else:
        return A