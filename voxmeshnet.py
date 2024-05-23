import trimesh
import torch, torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from itertools import chain
from nnspt.segmentation.unet import Unet

from layer_convertors import convert_inplace, LayerConvertorSm, LayerConvertorNNSPT

GraphConv = gnn.SAGEConv

class F2FBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(F2FBlock, self).__init__()

        self.downpath = nn.Linear(in_features, out_features)

        self.norm1 = nn.LayerNorm(out_features)
        self.act1 = nn.GELU()
        self.gconv1 = gnn.SAGEConv(out_features, out_features)

        self.norm2 = nn.LayerNorm(out_features)
        self.act2 = nn.GELU()
        self.gconv2 = gnn.SAGEConv(out_features, out_features)

        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edges):
        shortcut = self.shortcut(x)

        x = self.downpath(x)

        x = self.norm1(x)
        x = self.act1(x)
        x = self.gconv1(x, edges)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.gconv2(x, edges)

        x = x + shortcut

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

class Feature2VertexLayer(nn.Module):
    def __init__(self, in_features, hidden_layer_count):
        super(Feature2VertexLayer, self).__init__()

        self.gconv = []

        for i in range(hidden_layer_count, 1, -1):
            self.gconv += [GraphConv(i * in_features // hidden_layer_count, (i-1) * in_features // hidden_layer_count)]

        self.gconv_layer = nn.Sequential(*self.gconv)
        self.gconv_last = GraphConv(in_features // hidden_layer_count, 3)

    def forward(self, features, edges):
        for gconv_hidden in self.gconv:
            features = F.gelu(gconv_hidden(features, edges))

        return self.gconv_last(features, edges)

class SamplingBlock(nn.Module):
    def __init__(self, features_count, nneighbours=8):
        super(SamplingBlock, self).__init__()

        self.nneighbours = nneighbours

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, self.nneighbours+1), padding=0)

        self.shift_delta = nn.Conv1d(features_count, self.nneighbours*3, kernel_size=1, padding=0)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        self.feature_diff = nn.Linear(features_count + 3, features_count)
        self.feature_center = nn.Linear(features_count + 3, features_count)

    def forward(self, x, vertices):
        B, N, _ = vertices.shape

        points = vertices[:, :, None, None]
        xpoints = F.grid_sample(x, points, mode='bilinear', padding_mode='border', align_corners=True)
        cpoints = torch.cat([xpoints, points.permute(0, 4, 1, 2, 3)], dim=1)[:, :, :, :, 0]

        features = xpoints[:, :, :, 0, 0]

        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, self.nneighbours, 1, 3)
        neighbours = vertices[:, :, None, None] + shift_delta

        xneighbours = F.grid_sample(x, neighbours, mode='bilinear', padding_mode='border', align_corners=True)
        cneighbours = torch.cat([xneighbours, neighbours.permute(0, 4, 1, 2, 3)], dim=1)[:, :, :, :, 0]

        nfeatures = torch.cat([cpoints, cneighbours], dim=-1)

        nfeatures = nfeatures.permute([0, 3, 2, 1])
        nfeatures = self.feature_diff(nfeatures)
        nfeatures = nfeatures.permute([0, 3, 2, 1])

        nfeatures = self.sum_neighbourhood(nfeatures)[:, :, :, 0].transpose(2, 1)

        pfeatures =  cpoints[:, :, :, 0].transpose(2, 1)
        pfeatures = self.feature_center(pfeatures)

        features = pfeatures + nfeatures 
        return features

class VoxMeshNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, initial_mesh_path='./spheres/cube_3458_.obj'):
        super(VoxMeshNet, self).__init__()

        self.unet = Unet(in_channels=in_channels, out_channels=out_channels, encoder='timm-efficientnetv2-b1')

        convert_inplace(self.unet, LayerConvertorNNSPT)
        convert_inplace(self.unet, LayerConvertorSm)

        self.in_channels = in_channels
        self.out_channels = out_channels

        dim = 3
        steps = 4

        self.skip_count = self.unet.encoder.out_channels[:0:-1]
        self.latent_features_coount = [ 32 + 2 ** (steps-i) for i in range(steps+1)][::-1]

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []

        for i in range(steps+1):
            graph_unet_layers = []
            feature2vertex_layers = []
            skip = SamplingBlock(self.skip_count[i])

            for _ in range(self.out_channels-1):
                if i == 0:
                    graph_unet_layers += [ F2F(self.skip_count[i] + dim, self.latent_features_coount[i]) ]
                else:
                    graph_unet_layers += [ F2F(self.skip_count[i] + dim + self.latent_features_coount[i-1], self.latent_features_coount[i]) ]

                feature2vertex_layers.append(Feature2VertexLayer( self.latent_features_coount[i], 3))

            up_std_conv_layers.append(skip)
            up_f2f_layers.append(graph_unet_layers)
            up_f2v_layers.append(feature2vertex_layers)

        self.up_std_conv_layers = nn.ModuleList(up_std_conv_layers)
        self.up_f2f_layers = up_f2f_layers
        self.up_f2v_layers = up_f2v_layers

        self.decoder_f2f = nn.Sequential(*chain(*up_f2f_layers))
        self.decoder_f2v = nn.Sequential(*chain(*up_f2v_layers))

        with open(initial_mesh_path, 'r') as f:
            obj_dict = trimesh.exchange.obj.load_obj(f)

            vertices = obj_dict['vertices']
            faces = obj_dict['faces']

        vertices = torch.from_numpy(vertices).float()
        self.register_buffer('vertices', vertices)

        faces = torch.from_numpy(faces).long()[None]
        self.register_buffer('faces', faces)

        faces = self.faces[0]
        ffaces = torch.flip(faces, dims=(1, ))

        edges = torch.concatenate([
            faces[:,:2],
            faces[:,1:],
            faces[:,::2],
            ffaces[:,:2],
            ffaces[:,1:],
            ffaces[:,::2],
        ])

        edges = torch.unique(edges, dim=0).T
        self.register_buffer('edges', edges)

    def forward(self, x):
        vertices_ = self.vertices.clone().repeat(x.shape[0], 1, 1)
        faces = self.faces.clone().repeat(x.shape[0], 1, 1)

        edges = self.edges

        features = self.unet.encoder(x)
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = head

        pred = [None] * self.out_channels

        for k in range(self.out_channels-1):
            pred[k] = [[ vertices_.clone(), faces.clone(), None, None ]]

        for i, (sampling_block, up_f2f_layers, up_f2v_layers) in enumerate(zip(
            self.up_std_conv_layers, self.up_f2f_layers, self.up_f2v_layers
        )):
            block = self.unet.decoder.blocks[i]

            if i < len(skips) - 1:
                skip = skips[i]
                shape = skips[i].shape
            else:
                skip = None
                shape = skips[i].shape

            x = block(x, skip, shape)

            for k in range(self.out_channels-1):
                vertices = pred[k][i][0]
                faces = pred[k][i][1]
                latent_features = pred[k][i][2]
                f2f = up_f2f_layers[k]
                f2v = up_f2v_layers[k]

                skipped_features = sampling_block(x, vertices)

                if latent_features is not None:
                    latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2)
                else:
                    latent_features = torch.cat([skipped_features, vertices], dim=2)

                latent_features = f2f(latent_features, edges)
                features = f2v(latent_features, edges)
                vertices = torch.nn.functional.hardtanh(features)

                voxel_pred = self.unet.head(x) if i == len(self.up_std_conv_layers) - 1 else None

                pred[k] += [[vertices, faces, latent_features, voxel_pred]]

        return pred

