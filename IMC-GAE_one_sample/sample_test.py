import dgl
import torch as th
# 创建一个具有3种节点类型和3种边类型的异构图
graph_data = {
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}
g = dgl.heterograph(graph_data)
g.ntypes
g.etypes
g.canonical_etypes

neg_sampler = dgl.dataloading.negative_sampler.Uniform(4)
print(g)

g1 = neg_sampler(g, {("drug","interacts", "gene"): th.tensor([0])})

print(g1[('drug', 'interacts', 'gene')])

#dgl.add_edges, dgl.remove_edges

#g = dgl.add_edges(g,{("drug","interacts", "gene"): (th.tensor([0]),th.tensor([0]))})

#g = dgl.add_edges(g, {("drug","interacts", "gene"):（th.tensor([1, 3]), th.tensor([0, 1])})
g = dgl.add_edges(g, th.tensor([0]), th.tensor([3]), etype=('drug', 'interacts', 'gene'))
print(g)

#self.negatives = self.sample_negative(self.train_rating_info, self.sample_rate)