import torch
import torch_geometric as tg
from torch_geometric.nn import knn_graph
from dataloader import MPEGTransform, MPEGDataset, ADataListLoader
# for _ in range(100):
#     a = torch.randn([100,6])
#     b=torch.randn([9, 100, 6]) * 1e-7
#     c = [a]
#     for i in range(9):
#         print(b[i].norm())
#         c.append(c[-1] + b[i])
#     c = torch.cat(c, dim=0)
#     print(c.shape)
#     # 10 * 100
#     batch=torch.tensor([i for i in range(10) for _ in range(100)])
#     edge_index = knn_graph(c, k=16, batch=batch, loop=False)
#     print(edge_index, edge_index.shape)

#     row, col = edge_index
#     print((c[row] - c[col]).norm(dim=-1))
#     assert edge_index.shape[1] == 16000
dataset = MPEGDataset(root="data-0.50", pre_transform=MPEGTransform)
parallel = True
batch_size = 128
spec = dataset[13587]
print(spec.x)
print(knn_graph(spec.x, k=32, loop=False))
exit(0)
if parallel:
    train_loader = ADataListLoader(
        dataset,
        training=True,
        test_classes=[],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = ADataListLoader(
        dataset,
        training=False,
        test_classes=[0],
        batch_size=batch_size,
        shuffle=True,
    )
else:
    raise NotImplementedError