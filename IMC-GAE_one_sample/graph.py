import dgl
g = dgl.graph(([0, 1, 2], [1, 2, 3]))
neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
neg_sampler(g, [0, 1])