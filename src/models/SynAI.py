import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from timm.models.vision_transformer import Block
import timm


# #########################################################
# 
# Patch Embedders
# 
# #########################################################

class PatchEmbedder_wave2vec(nn.Module):
    def __init__(self, patch_size=32, in_channels=12, input_length=2560, embed_dim=384):
        super(PatchEmbedder_wave2vec, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.input_length = input_length
        self.embed_dim = embed_dim

        if input_length % patch_size:
            raise ValueError(f"input_length must be divided with patch_size, input_length:{input_length}, patch_size:{patch_size}")
        self.token_len = input_length // patch_size

        # patch_size에 따른 CNN 구성 정의
        configs = {
            8:  {"kernel": (10, 5), "stride": (4, 2)},
            16:  {"kernel": (10, 5, 3), "stride": (4, 2, 2)},
            32:  {"kernel": (10, 5, 3, 2), "stride": (4, 2, 2, 2)},
            64:  {"kernel": (10, 5, 3, 3, 2), "stride": (4, 2, 2, 2, 2)},
        }

        if patch_size not in configs:
            raise ValueError(f"Unsupported patch_size: {patch_size}")
        
        cfg = configs[patch_size]
        self.conv_kernel = cfg["kernel"]
        self.conv_stride = cfg["stride"]
        self.conv_dim = [embed_dim] * len(self.conv_kernel)

        # CNN layer 구성
        layers = []
        for k, s, d in zip(self.conv_kernel, self.conv_stride, self.conv_dim):
            layers.append(nn.Conv1d(in_channels, d, kernel_size=k, stride=s, padding=k // 3))
            layers.append(nn.GELU())
            in_channels = d
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 12, seq_length)
        B, C, T = x.shape
        x = self.encoder(x)  # (B, embed_dim, token_len)
        x = x.permute(0,2,1)   # (B, token_len, embed_dim)
        return x

class PatchEmbedder_resnet(nn.Module):  # only 32 patch만 지원
    class Bottleneck_1D(nn.Module):
        expansion = 4

        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()

            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)

            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)

            self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = F.relu(out)
            return out

    def __init__(self, patch_size=32, in_channels=12, input_length=2560, embed_dim=384):
        super().__init__()

        inplanes = embed_dim // patch_size
        self.in_channels = inplanes
        self.token_len = input_length // patch_size  # assumed sequence length
        ratio = [1, 2, 4, 8]
        layers=[6, 6, 6, 6]

        self.conv1 = nn.Conv1d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(int(inplanes * ratio[0]), layers[0], stride=1)
        self.layer2 = self._make_layer(int(inplanes * ratio[1]), layers[1], stride=2)
        self.layer3 = self._make_layer(int(inplanes * ratio[2]), layers[2], stride=2)
        self.layer4 = self._make_layer(int(inplanes * ratio[3]), layers[3], stride=2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.Bottleneck_1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * self.Bottleneck_1D.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.Bottleneck_1D.expansion)
            )

        layers = [self.Bottleneck_1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * self.Bottleneck_1D.expansion

        for _ in range(1, blocks):
            layers.append(self.Bottleneck_1D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.permute(0,2,1)  # (B, C, T) -> (B, T, C)

        return x
    
class ConcatProjectFusion(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, use_norm=True, dropout=0.1, hidden_ratio=1.0):
        """
        in_dim1: 첫 번째 feature의 embedding dim (e.g., wave2vec)
        in_dim2: 두 번째 feature의 embedding dim (e.g., resnet)
        out_dim: 최종 출력 dim (Transformer에 들어갈 embedding dim)
        use_norm: LayerNorm 사용할지 여부
        dropout: projection 후 dropout 비율
        hidden_ratio: MLP 중간 크기 (기본은 1배 = out_dim)
        """
        super().__init__()
        self.use_norm = use_norm

        concat_dim = in_dim1 + in_dim2
        hidden_dim = int(out_dim * hidden_ratio)

        self.proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

        if use_norm:
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, x1, x2):
        """
        x1, x2: (B, T, C1), (B, T, C2)
        returns: (B, T, out_dim)
        """
        x = torch.cat([x1, x2], dim=-1)  # (B, T, C1+C2)
        x = self.proj(x)  # (B, T, out_dim)
        if self.use_norm:
            x = self.norm(x)
        return x
    


class MAE_1D_250409_v3(nn.Module):  # No pretrained
    def __init__(
        self, 
        seq_length=2560,
        in_channels=12, 
        patch_size=32, 
        embed_dim=384,  # head 갯수로 나눠져야함
        
        merge_mode = 'projection',  # linear_projection avg add
        encoder = 'vit_encoder',

        decoder_depth=2, 
        decoder_num_heads=8,
        stft_loss_ratio=0,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.decoder_embed_dim = embed_dim
        self.stft_loss_ratio = stft_loss_ratio
        self.merge_mode = merge_mode


        ### Patch embedding ###
        self.patch_embed_wave2vec = PatchEmbedder_wave2vec(
            patch_size=patch_size, in_channels=in_channels, input_length=seq_length, embed_dim=embed_dim
        )
        self.patch_embed_resnet = PatchEmbedder_resnet(
            patch_size=patch_size, in_channels=in_channels, input_length=seq_length, embed_dim=embed_dim
        )
        self.token_len = self.patch_embed_wave2vec.token_len
        self.encoder_pos = nn.Parameter(torch.zeros(1, self.token_len, embed_dim))
        self.fusion = ConcatProjectFusion(in_dim1=embed_dim, in_dim2=embed_dim, out_dim=embed_dim)


        ### Transformer Encoder Blocks ###
        if encoder == 'vit_encoder':
            if embed_dim == 384:
                vit = timm.create_model('vit_small_patch16_224', pretrained=False)
            elif embed_dim == 768:
                vit = timm.create_model('vit_base_patch16_224', pretrained=False)
            elif embed_dim == 1024:
                vit = timm.create_model('vit_large_patch16_224', pretrained=False)
            encoder_ = vit.blocks  # nn.Sequential([...])
            for param in encoder_.parameters():
                param.requires_grad = True            
        self.encoder = encoder_  # nn.Sequential([...])
        self.encoder_norm = nn.LayerNorm(embed_dim)


        ### MAE decoder specifics ###
        self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.token_len, self.decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(
                self.decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, 
                qk_norm=None, norm_layer=nn.LayerNorm
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Sequential(
            nn.Linear(self.decoder_embed_dim, patch_size * in_channels, bias=True),
            nn.Tanh()            
        )

        # print(f'seq_length {seq_length}, in_channels {in_channels}, patch_size {patch_size}, embed_dim {embed_dim}, token_len {self.token_len}, ')

        # 가중치 초기화
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.encoder_pos, std=.02)
        nn.init.normal_(self.decoder_pos, std=.02)
        self.apply(self._init_weights)  # Linear, LayerNorm 초기화

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def local_masking(self, x, mask_ratio, mask_token=None):
        """
        x: (B, T, C)
        mask_token: (1, 1, C) or (1, C)
        """
        B, T, C = x.shape
        len_keep = int(T * (1 - mask_ratio))

        noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        # 1. x_masked: 복원용 encoder 입력
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))

        # 2. mask: 0=keep, 1=masked
        mask = torch.ones(B, T, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # (B, T)

        # 3. x_full_with_mask: masked position은 mask_token으로 대체
        if mask_token is None:
            mask_token = torch.zeros(1, 1, C, device=x.device)

        # mask를 (B, T, C)로 확장
        mask_expanded = mask.unsqueeze(-1).expand(B, T, C)  # (B, T, 1) -> (B, T, C)
        mask_tokens = mask_token.expand(B, T, C)
        
        x_full_with_mask = x.clone()
        x_full_with_mask = torch.where(mask_expanded.bool(), mask_tokens, x_full_with_mask)

        return x_masked, x_full_with_mask, mask, ids_restore, ids_keep

    def patchify(self, x):
        """
        x: (N, C, L)
        """
        N, C, L = x.shape
        x = x.reshape(N, C, self.token_len, self.patch_size)
        x = x.permute(0, 2, 3, 1)  # (N, num_patches, p, C)
        x = x.reshape(N, self.token_len, self.patch_size * C)

        return x
        
    def forward_encoder(self, x, mask_ratio=0):
        '''
        x.shape = (B, in_channel, seq_length)
        '''
        x1 = self.patch_embed_wave2vec(x)  # (B, T, C)
        x2 = self.patch_embed_resnet(x)  # (B, T, C)
        if self.merge_mode == 'projection':
            x = self.fusion(x1, x2)  # (B, T, C)
        elif self.merge_mode == 'avg':
            x = 0.5 * (x1 + x2)
        elif self.merge_mode == 'add':
            x = x1 + x2
        x = x + self.encoder_pos  # (B, T, C)
        B, T, C = x.shape
        
        if mask_ratio > 0: 
            x, x_full_with_mask, mask, ids_restore, ids_keep = self.local_masking(x, mask_ratio=mask_ratio)
        else:
            x_full_with_mask = x
            mask = torch.zeros(B, T, device=x.device)  # 모든 토큰이 유지됨을 나타냄    
            ids_restore = torch.arange(T, device=x.device).expand(B, T)  # 순서 그대로 유지            

        x = self.encoder(x)   # (B, T, C)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        B, _, embed_dim = x.shape
        x = self.decoder_embed(x)

        batch_size = x.shape[0]
        seq_len = ids_restore.shape[1]
        mask_tokens = self.mask_token.repeat(batch_size, seq_len - x.shape[1], 1) # (N, len_mask, 1)
        
        x = torch.cat([x, mask_tokens], dim=1)  # (N, num_patches, embed_dim)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # 토큰들을 원래 순서로 재배열 (unshuffle)
        x = x + self.decoder_pos
    
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (B, token_len, in_channels * patch_size)

        x = x.view(B, self.token_len, self.in_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (B, in_channels, token_len_per_ch, patch_size)
        x = x.reshape(B, self.in_channels, self.token_len * self.patch_size)  # (B, in_channels, seq_length?)
        
        return x
    
    @staticmethod
    def stft_combined_loss(pred, label, n_fft=256, hop_length=None, alpha=0.5):
        """
        STFT 기반 Magnitude + Phase 손실 함수
        pred: (batch, channels, time) - 예측된 신호
        label: (batch, channels, time) - 원본 신호
        n_fft: FFT 창 크기
        hop_length: 창 이동 크기 (기본값: n_fft//2)
        alpha: Magnitude vs Phase Loss 비율 (0~1 사이)
        """

        if hop_length is None:
            hop_length = n_fft // 2  # 기본값 설정

        B, C, T = pred.shape
        pred = pred.reshape(B * C, T)
        label = label.reshape(B * C, T)
        
        # 🔹 STFT 변환 (복소수 형태)
        pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        label_stft = torch.stft(label, n_fft=n_fft, hop_length=hop_length, return_complex=True)

        # 🔹 Magnitude(진폭) 손실 (log1p 적용)
        magnitude_loss = F.mse_loss(torch.log1p(torch.abs(pred_stft)), torch.log1p(torch.abs(label_stft)))

        # 🔹 Phase(위상) 손실 (sin 변환 적용)
        pred_phase = torch.angle(pred_stft)
        label_phase = torch.angle(label_stft)
        phase_loss = F.mse_loss(torch.sin(pred_phase), torch.sin(label_phase))

        # 🔹 최종 손실 (가중 조합)
        loss = alpha * magnitude_loss + (1 - alpha) * phase_loss

        return loss, magnitude_loss, phase_loss

    def forward_loss(self, imgs, pred, mask):
        """
        imgs, pred: (B, in_channels, seq_length)
        mask: (B, in_channels, seq_length) - 0=keep, 1=masked
        """
        # 전체 복원 손실 (MSE)
        mse_loss_full = ((pred - imgs) ** 2).mean()

        # mask된 부분만 추출하여 복원 손실 계산
        masked_pred = pred * mask  # mask가 1인 부분만 남음
        masked_imgs = imgs * mask
        masked_mse_loss = ((masked_pred - masked_imgs) ** 2).sum() / (mask.sum() + 1e-8)  # mask된 영역의 평균 손실

        # 최종 손실: mask된 부분 손실 + 전체 손실
        alpha = 0.75
        loss = alpha * masked_mse_loss + (1-alpha) * mse_loss_full

        # STFT 손실이 있는 경우 추가
        if self.stft_loss_ratio > 0:
            stft_loss, _, _ = self.stft_combined_loss(pred, imgs)
            loss += self.stft_loss_ratio * stft_loss

        return loss

    def forward(self, x, mask_ratio=None):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        
        # mask 준비
        mask = mask.unsqueeze(1).repeat(1, self.in_channels, 1)  # (B, 12, token_len)
        mask = mask.repeat_interleave(self.patch_size, dim=-1)   # (B, 12, seq_length) 
        
        # 손실 계산
        loss = self.forward_loss(x, pred, mask)

        return loss, pred, mask




class MLPHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(embed_dim))  # 안정성 확보
            layers.append(nn.Dropout(dropout))  # Dropout 추가
        layers.append(nn.Linear(embed_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        # 안정적인 초기화
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.mlp(x)
    
class OnlyEncoderForFT_250409(nn.Module):
    def __init__(
            self, mae_model, num_classes, embed_dim,
        ):
        super().__init__()

        self.mae_model = mae_model        
        self.patch_embed_wave2vec = mae_model.patch_embed_wave2vec
        self.patch_embed_resnet = mae_model.patch_embed_resnet
        self.encoder_pos = mae_model.encoder_pos
        self.encoder = mae_model.encoder
        self.encoder_norm = mae_model.encoder_norm
        self.fusion = mae_model.fusion
        self.merge_mode = mae_model.merge_mode
        self.embed_dim = embed_dim

        # Downstream Head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token_pos, std=0.02)
        self.head = MLPHead(
            embed_dim=embed_dim, num_classes=num_classes, hidden_layers=3, dropout=0.1
        )

    def forward(self, x):
        """
        x: (B, in_channels, seq_length)
        """
        x1 = self.patch_embed_wave2vec(x)  # (B, T, C)
        x2 = self.patch_embed_resnet(x)  # (B, T, C)
        if self.merge_mode == 'projection':
            x = self.fusion(x1, x2)  # (B, T, C)
        elif self.merge_mode == 'avg':
            x = 0.5 * (x1 + x2)
        elif self.merge_mode == 'add':
            x = x1 + x2
        x = x + self.encoder_pos  # (B, T, C)
        B, T, C = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        cls_token_pos = self.cls_token_pos.expand(B, -1, -1)
        cls_token = cls_token + cls_token_pos
        x = torch.cat([cls_token, x], dim=1)

        x = self.encoder(x)
        x = self.encoder_norm(x)

        cls_feature = x[:, 0, :]  # (B, embed_dim)
        global_features = x[:, 1:, :]
    
        logits = self.head(cls_feature)
        logits = torch.clamp(logits, min=-10, max=10)

        return logits
    

def Build_Model_250526(model_info):
    if model_info['name'] == 'MAE_1D_250409_v3':
        model_class = MAE_1D_250409_v3
    else:
        raise ValueError(f"Unknown model name : {model_info['name']}")
    print(f"model name : {model_info['name']}")
    
    if model_info['name'] == 'resnet':
        model = model_class(
            depth = model_info['config']['depth'],
            input_length = model_info['config']['seq_length'],
            in_channels = model_info['config']['in_channels'],
            num_classes = model_info['config']['num_classes'],
        )
        return model
    
    elif model_info['name'] == 'vit':
        model = model_class(
            seq_length = model_info['config']['seq_length'],
            in_channels = model_info['config']['in_channels'],
            patch_size = model_info['config']['patch_size'],
            embed_dim = model_info['config']['embed_dim'],
            num_classes = model_info['config']['num_classes'],                        
            dropout = model_info['config']['dropout'],                        
        )
        return model

    else:
        model = model_class(
            seq_length=model_info['config']['seq_length'],
            in_channels = model_info['config']['in_channels'],
            patch_size = model_info['config']['patch_size'],
            embed_dim = model_info['config']['embed_dim'],
            merge_mode = model_info['config']['merge_mode'],  # linear_projection avg add
            encoder = model_info['config']['encoder'],
            # for self-supervised learning
            decoder_depth = model_info['config']['decoder_depth'],
            decoder_num_heads = model_info['config']['decoder_num_heads'],
            stft_loss_ratio = model_info['config']['stft_loss_ratio'],   
        )
        if model_info['mode'] == 'pretraining':
            return model
        
        elif model_info['mode'] == 'finetuning':
            if model_info['weights_init'] == 'SSL_transfer':
                checkpoint = torch.load(model_info['prev_model_path'], weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"model was loaded from '{model_info['prev_model_path']}'")

            model = OnlyEncoderForFT_250409(
                model, 
                num_classes = model_info['config']['num_classes'], 
                embed_dim = model_info['config']['embed_dim'],
            )

            if model_info['weights_init'] == 'DST_transfer':
                checkpoint = torch.load(model_info['prev_model_path'], weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"model was loaded from '{model_info['prev_model_path']}'")
                return model
            elif model_info['weights_init'] == 'scratch':
                print(f"model was loaded from scratch")
                return model
            else:
                return model


# def Build_Model_250526_v2(model_info):
#     if model_info['name'] == 'MAE_1D_250409_v3':
#         model_class = MAE_1D_250409_v3
#     else:
#         raise ValueError(f"Unknown model name : {model_info['name']}")
#     print(f"model name : {model_info['name']}")
    
#     if model_info['name'] == 'resnet':
#         model = model_class(
#             depth = model_info['config']['depth'],
#             input_length = model_info['config']['seq_length'],
#             in_channels = model_info['config']['in_channels'],
#             num_classes = model_info['config']['num_classes'],
#         )
#         return model
    
#     elif model_info['name'] == 'vit':
#         model = model_class(
#             seq_length = model_info['config']['seq_length'],
#             in_channels = model_info['config']['in_channels'],
#             patch_size = model_info['config']['patch_size'],
#             embed_dim = model_info['config']['embed_dim'],
#             num_classes = model_info['config']['num_classes'],                        
#             dropout = model_info['config']['dropout'],                        
#         )
#         return model

#     else:
#         model = model_class(
#             seq_length=model_info['config']['seq_length'],
#             in_channels = model_info['config']['in_channels'],
#             patch_size = model_info['config']['patch_size'],
#             embed_dim = model_info['config']['embed_dim'],
#             merge_mode = model_info['config']['merge_mode'],  # linear_projection avg add
#             encoder = model_info['config']['encoder'],
#             # for self-supervised learning
#             decoder_depth = model_info['config']['decoder_depth'],
#             decoder_num_heads = model_info['config']['decoder_num_heads'],
#             stft_loss_ratio = model_info['config']['stft_loss_ratio'],   
#         )
#         if model_info['mode'] == 'pretraining':
#             return model
        
#         elif model_info['mode'] == 'finetuning':
#             if model_info['weights_init'] == 'SSL_transfer':
#                 checkpoint = torch.load(model_info['prev_model_path'], weights_only=True)
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 print(f"model was loaded from '{model_info['prev_model_path']}'")

#             model = OnlyEncoderForFT_250409(model, 
#                 num_classes = model_info['config']['num_classes'], 
#                 embed_dim = model_info['config']['embed_dim'],
#             )

#             if model_info['weights_init'] == 'DST_transfer':
#                 model = OnlyEncoderForFT_250409(model, 
#                     num_classes = model_info['config']['num_classes'], 
#                     embed_dim = model_info['config']['embed_dim'],
#                 )

#                 checkpoint = torch.load(model_info['prev_model_path'], weights_only=True)
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 print(f"model was loaded from '{model_info['prev_model_path']}'")
#                 return model
#             elif model_info['weights_init'] == 'scratch':
#                 print(f"model was loaded from scratch")
#                 return model
#             else:
#                 return model