import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from clip import clip
from .nn import timestep_embedding


class CLIP_gd(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.finetune_clip_layer = getattr(args, 'finetune_clip_layer', 'all')
        clip_model, preprocess = clip.load('RN50x16', jit=False)
        self.preprocess = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        clip_model = clip_model.float()

        # visual
        self.visual_frozen = nn.Sequential(
            clip_model.visual.conv1,
            clip_model.visual.bn1,
            clip_model.visual.relu1,
            clip_model.visual.conv2,
            clip_model.visual.bn2,
            clip_model.visual.relu2,
            clip_model.visual.conv3,
            clip_model.visual.bn3,
            clip_model.visual.relu3,
            clip_model.visual.avgpool,
            clip_model.visual.layer1,
            clip_model.visual.layer2,
            clip_model.visual.layer3,
        )
        self.attn_pool = clip_model.visual.attnpool
        self.layer4 = clip_model.visual.layer4

        self.attn_resolution = args.image_size // 32
        self.image_size = args.image_size
        self.after_load()
        self.define_finetune()

    def after_load(self):
        self.attn_pool.positional_embedding = nn.Parameter(torch.randn(self.attn_resolution ** 2 + 1, 3072) / 3072 ** 0.5)
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        emb_dim = [48, 48, 96]
        tmp1 = [96, 96, 384]
        for cnt in range(6):
            emb_dim.extend(tmp1)
        tmp2 = [192, 192, 768]
        for cnt in range(8):
            emb_dim.extend(tmp2)
        tmp3 = [384, 384, 1536]
        for cnt in range(18):
            emb_dim.extend(tmp3)
        tmp4 = [768, 768, 3072]
        for cnt in range(8):
            emb_dim.extend(tmp4)
        self.emb_layers = nn.Sequential(nn.ReLU(), nn.Linear(512, sum(emb_dim) * 2))
        self.split_idx = []
        cur_idx = 0
        for cnt in range(len(emb_dim)):
            self.split_idx.append(cur_idx + emb_dim[cnt])
            self.split_idx.append(cur_idx + 2 * emb_dim[cnt])
            cur_idx += 2 * emb_dim[cnt]
        self.split_idx = self.split_idx[:-1]

    def define_finetune(self):
        self.train()

        # freeze visual encoder
        for param in self.visual_frozen.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        self.attn_pool.positional_embedding.requires_grad = True
        self.time_embed.requires_grad = True
        self.emb_layers.requires_grad = True
        
        if self.finetune_clip_layer == 'last':
            for param in self.layer4.parameters():
                param.requires_grad = True
            for param in self.attn_pool.parameters():
                param.requires_grad = True
        elif self.finetune_clip_layer == 'all':
            for param in self.parameters():
                param.requires_grad = True


    def train(self, mode=True):
        self.visual_frozen.eval()
        self.layer4.eval()
        self.attn_pool.eval()

        self.time_embed.train(mode)
        self.emb_layers.train(mode)

        if self.finetune_clip_layer == 'last':
            self.layer4.train(mode)
            self.attn_pool.train(mode)
        elif self.finetune_clip_layer == 'all':
            self.visual_frozen.train(mode)
            self.layer4.train(mode)
            self.attn_pool.train(mode)

    def encode_image(self, image, t):
        image = (image + 1) / 2.0
        image = self.preprocess(image)
        emb = self.time_embed(timestep_embedding(t, 128))
        emb_out = torch.tensor_split(self.emb_layers(emb).unsqueeze(-1).unsqueeze(-1), self.split_idx, dim=1)
        x = self.visual_frozen[1](self.visual_frozen[0](image))
        x = x * (1 + emb_out[0]) + emb_out[1]
        x = self.visual_frozen[4](self.visual_frozen[3](self.visual_frozen[2](x)))
        x = x * (1 + emb_out[2]) + emb_out[3]
        x = self.visual_frozen[7](self.visual_frozen[6](self.visual_frozen[5](x)))
        x = x * (1 + emb_out[4]) + emb_out[5]
        x = self.visual_frozen[9](self.visual_frozen[8](x))
        # layer1
        module_cnt = 10
        emb_cnt = 6
        for cnt in range(6):
            x = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x, emb_out, module_cnt, emb_cnt, cnt)

        # layer2
        module_cnt = 11
        emb_cnt = 42
        for cnt in range(8):
            x = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x, emb_out, module_cnt, emb_cnt, cnt)
        # layer3
        module_cnt = 12
        emb_cnt = 90
        for cnt in range(18):
            x = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x, emb_out, module_cnt, emb_cnt, cnt)

        # layer4
        emb_cnt = 198
        for cnt in range(8):
            x = self.bottleneck_block_forward(self.layer4[cnt], x, emb_out, module_cnt, emb_cnt, cnt)

        x = self.attn_pool(x)

        return x

    def encode_image_list(self, image, t, return_layer=4):
        image = (image + 1) / 2.0
        image = self.preprocess(image)

        emb = self.time_embed(timestep_embedding(t, 128))
        emb_out = torch.tensor_split(self.emb_layers(emb).unsqueeze(-1).unsqueeze(-1), self.split_idx, dim=1)
        x = self.visual_frozen[1](self.visual_frozen[0](image))
        x = x * (1 + emb_out[0]) + emb_out[1]
        x = self.visual_frozen[4](self.visual_frozen[3](self.visual_frozen[2](x)))
        x = x * (1 + emb_out[2]) + emb_out[3]
        x = self.visual_frozen[7](self.visual_frozen[6](self.visual_frozen[5](x)))
        x = x * (1 + emb_out[4]) + emb_out[5]
        x1 = self.visual_frozen[9](self.visual_frozen[8](x))
        # layer1
        module_cnt = 10
        emb_cnt = 6

        for cnt in range(6):
            if cnt == 0:
                x2 = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x1, emb_out, module_cnt, emb_cnt, cnt)
            else:
                x2 = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x2, emb_out, module_cnt, emb_cnt, cnt)
        
        # layer2
        module_cnt = 11
        emb_cnt = 42
        for cnt in range(8):
            if cnt == 0:
                x3 = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x2, emb_out, module_cnt, emb_cnt, cnt)
            else:
                x3 = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x3, emb_out, module_cnt, emb_cnt, cnt)

        # layer3
        module_cnt = 12
        emb_cnt = 90
        for cnt in range(18):
            if cnt == 0:
                x4 = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x3, emb_out, module_cnt, emb_cnt, cnt)
            else:
                x4 = self.bottleneck_block_forward(self.visual_frozen[module_cnt][cnt], x4, emb_out, module_cnt, emb_cnt, cnt)

        # layer4
        emb_cnt = 198
        for cnt in range(8):
            if cnt == 0:
                x5 = self.bottleneck_block_forward(self.layer4[cnt], x4, emb_out, module_cnt, emb_cnt, cnt)
            else:
                x5 = self.bottleneck_block_forward(self.layer4[cnt], x5, emb_out, module_cnt, emb_cnt, cnt)

        x6 = self.attn_pool(x5)

        return [x1, x2, x3, x4, x5, x6]



    def bottleneck_block_forward(self, net, x, emb_out, module_cnt, emb_cnt, cnt):
        identity = x
        y = net.bn1(net.conv1(x))
        y = y * (1 + emb_out[emb_cnt+cnt*6]) + emb_out[emb_cnt+cnt*6+1]
        y = net.relu1(y)
        y = net.bn2(net.conv2(y))
        y = y * (1 + emb_out[emb_cnt+cnt*6+2]) + emb_out[emb_cnt+cnt*6+3]
        y = net.relu2(y)
        y = net.avgpool(y)
        y = net.bn3(net.conv3(y))
        y = y * (1 + emb_out[emb_cnt+cnt*6+4]) + emb_out[emb_cnt+cnt*6+5]
        if net.downsample is not None:
            identity = net.downsample(x)
        y += identity
        y = net.relu3(y)
        return y


    def encode_text(self, text):
        with torch.no_grad():
            x = self.token_embedding_frozen(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding_frozen
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer_frozen(x)

        x = self.transformer_last_block(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def unfreeze(self):
        self.attn_pool.requires_grad_(True)
        self.layer4.requires_grad_(True)

        self.transformer_last_block.requires_grad_(True)
        self.ln_final.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        self.logit_scale.requires_grad_(True)

    def forward(self, image, text_features, timesteps):
        # to match the preprocess of clip model

        image_features = self.encode_image(image, timesteps)
        #text_features = self.encode_text(text)

        return image_features, text_features

    def training_step(self, batch, batch_idx):
        image, text = batch

        bs = image.size(0)

        image_features, text_features = self(image, text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        label = torch.arange(bs).long()
        label = label.to(image.device)

        loss_i = F.cross_entropy(logits_per_image, label)
        loss_t = F.cross_entropy(logits_per_text, label)

        loss = (loss_i + loss_t) / 2

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(list(self.attn_pool.parameters()) +
                                list(self.layer4.parameters()) +
                                list(self.transformer_last_block.parameters()) +
                                list(self.ln_final.parameters()) +
                                [self.text_projection],
                                lr=lr)
        return opt

