import trimesh
import torch, torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from itertools import chain
from nnspt.segmentation.unet import Unet

from layer_convertors import convert_inplace, LayerConvertorSm, LayerConvertorNNSPT

class F2FBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(F2FBlock, self).__init__()

        self.act = nn.GELU()

        self.downpath = nn.Linear(in_features, out_features)

        self.gconv1 = gnn.SAGEConv(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)

        self.gconv2 = gnn.SAGEConv(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edges):
        shortcut = self.shortcut(x)

        x = self.downpath(x)
        x = self.act(x)

        x = self.gconv1(x, edges)
        x = self.norm1(x)
        x = self.act(x)

        x = self.gconv2(x, edges)
        x = self.norm2(x)

        x = x + shortcut
        x = self.act(x)

        return x

class F2F(nn.Module):
    def __init__(self, in_features, out_features):
        super(F2F, self).__init__()

        blocks = [
            F2FBlock(in_features, out_features),
            F2FBlock(out_features, out_features),
            F2FBlock(out_features, out_features),
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, features, edges):
        for block in self.blocks:
            features = block(features, edges)

        return features

class F2V(nn.Module):
    def __init__(self, in_features, hidden_layer_count):
        super(F2V, self).__init__()

        gconv = []

        for i in range(hidden_layer_count, 1, -1):
            gconv.append(
                gnn.SAGEConv(
                    i * in_features // hidden_layer_count,
                   (i - 1) * in_features // hidden_layer_count
                )
            )

        self.gconv = nn.Sequential(*gconv)
        self.gconv_last = gnn.SAGEConv(in_features // hidden_layer_count, 3)

    def forward(self, features, edges):
        for gconv_hidden in self.gconv:
            features = F.gelu(gconv_hidden(features, edges))

        return self.gconv_last(features, edges)

class SamplingBlock(nn.Module):
    def __init__(self, nfeatures, nneighbours=8):
        super(SamplingBlock, self).__init__()

        self.ndim = 3

        self.nshape = 4
        self.shape_head = nn.Linear(4, self.nshape)

        self.nneighbours = nneighbours

        self.delta = nn.Conv1d(nfeatures+self.ndim+self.nshape, self.nneighbours*self.ndim, kernel_size=1, padding=0)
        self.delta.weight.data.fill_(0.0)

        self.point_head = nn.Sequential(
            nn.Conv2d(nfeatures+self.ndim, nfeatures, kernel_size=1)
        )

        self.neighbour_head = nn.Sequential(
            nn.Conv2d(nfeatures+self.ndim, nfeatures, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(nfeatures, nfeatures, kernel_size=(1, self.nneighbours+1), padding=0, bias=False),
        )

    def forward(self, x, vertices):
        B, N, _ = vertices.shape

        points = vertices[:, :, None, None]
        xpoints = F.grid_sample(x, points, mode='bilinear', padding_mode='border', align_corners=True)
        cpoints = torch.cat([xpoints, points.permute(0, 4, 1, 2, 3)], dim=1)[:, :, :, :, 0]

        pfeatures = self.point_head(cpoints)[:, :, :, 0].transpose(2, 1)

        features = cpoints[:, :, :, 0]

        shape = torch.tensor(x.shape[1:], dtype=torch.float).to(x.device)
        shape = shape[None].repeat(B, 1)

        sfeatures = self.shape_head(shape)[:, :, None].repeat(1, 1, N)
        features = torch.cat([features, sfeatures], dim=1)

        delta = self.delta(features).permute(0, 2, 1).view(B, N, self.nneighbours, 1, self.ndim)
        neighbours = vertices[:, :, None, None] + delta

        xneighbours = F.grid_sample(x, neighbours, mode='bilinear', padding_mode='border', align_corners=True)
        cneighbours = torch.cat([xneighbours, neighbours.permute(0, 4, 1, 2, 3)], dim=1)[:, :, :, :, 0]

        nfeatures = torch.cat([cpoints, cneighbours], dim=-1)
        nfeatures = self.neighbour_head(nfeatures)[:, :, :, 0].transpose(2, 1)
        features = pfeatures + nfeatures

        return features

class GeneratingBlock(nn.Module):
    def __init__(self, nfeatures, npoints=3458):
        super(GeneratingBlock, self).__init__()

        self.nshape = 4
        self.sproj = nn.Linear(4, self.nshape)

        self.npoints = npoints

        base = int(self.npoints ** 0.3)
        base3d = base ** 3

        self.base3d = base3d

        self.vproj = nn.Sequential(
            nn.AdaptiveAvgPool3d((base, base, base)),
            nn.Flatten(start_dim=2),
            nn.Linear(base3d, npoints),
        )

        self.nfeatures = nfeatures

        self.proj = nn.Sequential(
            nn.Linear(nfeatures + self.nshape, nfeatures),
        )

    def forward(self, x, vertices):
        B, N, _ = vertices.shape

        shape = torch.tensor(x.shape[1:], dtype=torch.float).to(x.device)
        shape = shape[None].repeat(B, 1)

        s = self.sproj(shape)[:, :, None].repeat(1, 1, N)
        x = self.vproj(x)

        x = torch.concatenate([x, s], dim=1)
        x = x.transpose(2, 1)

        x = self.proj(x)

        return x

class VoxMeshNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            out_meshes=1,
            depth=5,
            initial_mesh_path='./spheres/cube_3458_.obj',
            nneighbours=8
        ):
        """
            :args:
                in_channels: int
                    ...
                out_channels: int
                    ...
                out_meshes: int
                    ...
                depth: int
                    ...
                initial_mesh_path: str
                    ...
                nneighbours: int
                    ...
        """
        super(VoxMeshNet, self).__init__()

        self.depth = depth

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_meshes = out_meshes

        self.load_initial_mesh(initial_mesh_path)

        unet = Unet(in_channels=in_channels, out_channels=out_channels, encoder='timm-efficientnetv2-b1', depth=depth)

        convert_inplace(unet, LayerConvertorNNSPT)
        convert_inplace(unet, LayerConvertorSm)

        self.encoder = unet.encoder
        self.voxel_decoder = unet.decoder

        self.segmentation_head = unet.head

        self.skip_count = self.encoder.out_channels[:0:-1]
        self.lfeature_count = [ 32 + 2 ** (self.depth-i) for i in range(self.depth) ][::-1]

        sampling_blocks = []
        f2f_layers = []
        f2v_layers = []

        for i in range(self.depth):
            if i == 0:
                sampling_block = GeneratingBlock(self.skip_count[i], self.nvertices)
            else:
                sampling_block = SamplingBlock(self.skip_count[i], nneighbours)

            for _ in range(self.out_meshes):
                if i == 0:
                    in_features = self.skip_count[i] + self.ndim
                else:
                    in_features = self.skip_count[i] + self.ndim + self.lfeature_count[i-1]

                f2f_layer = F2F(
                    in_features,
                    self.lfeature_count[i]
                )

                f2v_layer = F2V(
                    self.lfeature_count[i],
                    self.ndim
                )

            sampling_blocks.append(sampling_block)
            f2f_layers.append(f2f_layer)
            f2v_layers.append(f2v_layer)

        self.sampling_blocks = nn.ModuleList(sampling_blocks)
        self.f2f_layers = nn.ModuleList(f2f_layers)
        self.f2v_layers = nn.ModuleList(f2v_layers)

    def load_initial_mesh(self, initial_mesh_path):
        with open(initial_mesh_path, 'r') as f:
            obj_dict = trimesh.exchange.obj.load_obj(f)

            vertices = obj_dict['vertices']
            faces = obj_dict['faces']

        vertices = torch.from_numpy(vertices).float()
        self.register_buffer('vertices', vertices)

        self.nvertices, self.ndim = vertices.shape

        faces = torch.from_numpy(faces).long()[None]
        self.register_buffer('faces', faces)

        faces = self.faces[0]
        ffaces = torch.flip(faces, dims=(1, ))

        edges = torch.concatenate([
            faces[:, :2],
            faces[:, 1:],
            faces[:, ::2],
            ffaces[:, :2],
            ffaces[:, 1:],
            ffaces[:, ::2],
        ])

        edges = torch.unique(edges, dim=0).T
        self.register_buffer('edges', edges)

    @staticmethod
    def denormalize_vertices(vertices):
        return vertices

    def forward(self, x):
        vertices_ = self.vertices.clone().repeat(x.shape[0], 1, 1)
        faces = self.faces.clone().repeat(x.shape[0], 1, 1)

        vfeatures = self.encoder(x)
        vfeatures = vfeatures[::-1]

        head = vfeatures[0]
        vskips = vfeatures[1:]

        x = head

        pred = [None] * self.out_channels

        for k in range(self.out_meshes):
            pred[k] = [[ self.denormalize_vertices(vertices_.clone()), faces.clone(), None ]]

        for i in range(self.depth):
            vblock = self.voxel_decoder.blocks[i]
            mblock = self.sampling_blocks[i]

            f2f = self.f2f_layers[i]
            f2v = self.f2v_layers[i]

            if i < len(vskips) - 1:
                vskip = vskips[i]
                vshape = vskips[i].shape
            else:
                vskip = None
                vshape = vskips[i].shape

            x = vblock(x, vskip, vshape)


            for k in range(self.out_meshes):
                vertices = pred[k][i][0]
                latent_features = pred[k][i][2]

                xpoints = mblock(x, vertices)

                if latent_features is not None:
                    latent_features = torch.cat([latent_features, xpoints, vertices], dim=2)
                else:
                    latent_features = torch.cat([xpoints, vertices], dim=2)

                latent_features = f2f(latent_features, self.edges)
                features = f2v(latent_features, self.edges)
                vertices = torch.nn.functional.hardtanh(features)

                pred[k] += [[
                    self.denormalize_vertices(vertices),
                    faces,
                    latent_features
                ]]

        pred[0][-1].append(self.segmentation_head(x))

        return pred
