import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SimWithCenter(nn.Module):
    def __init__(self, dim_output):
        super(SimWithCenter, self).__init__()
        self.linear = nn.Linear(dim_output, 1)
        self._init_params()

    def forward(self, x):
        B = x.shape[0]
        x = torch.sigmoid(x) 
        x = self.linear(x.view(B, -1))
        x = torch.sigmoid(x) 
        x = x.view(B) 
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def get_sim_matrix(img_patch_feat, text_feat, target_area, f_shape):
    B, N, D = img_patch_feat.shape
        
    if text_feat.dim() == 2:
        text_feat = text_feat.unsqueeze(1).expand(-1, N, -1)
            
    patch_pred = torch.sum(img_patch_feat * text_feat, dim=-1)
    H, W = target_area
    patch_grid = patch_pred.view(B, 1, f_shape[0], f_shape[1])
    pixel_pred = F.interpolate(patch_grid, size=(H, W), mode='bilinear', align_corners=False)
    pixel_pred = pixel_pred.squeeze(1)  # [B, H, W]
        
    return pixel_pred


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
import clip.clip as clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, tokens, i):
        tokens = self.fc[i](tokens)
        return tokens
    

class TransReID(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(TransReID, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.linear = nn.Linear(self.in_planes_proj, self.in_planes_proj+self.in_planes, bias=False)
        
        self.classifier = nn.Linear(self.in_planes+self.in_planes_proj, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)


        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        
        ##### FOR CLIP VIT
        self.image_encoder = clip_model.visual
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        self.prompt_learner = PromptLearner(clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.text_encoder.eval()
        self.sim_with_center = SimWithCenter(cfg.INPUT.SIZE_TRAIN[1]*cfg.INPUT.SIZE_TRAIN[0])

        scale = self.in_planes ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(self.in_planes, self.in_planes_proj))


    def forward(self, x, cam_label= None, view_label=None, get_image=None, get_matrix = True):
        #### clip vit
        if cam_label != None and view_label!=None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif view_label!=None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        image_features_11, image_features, image_features_proj, _ = self.image_encoder(x, cv_embed)

        img_feature = image_features[:,0]
        img_feature_proj = image_features_proj[:,0]
        img_feat_proj_patch = image_features_proj[:,1:]
        img_feat_patch_11 = image_features_11[:,1:]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        out_feat = torch.cat([feat, feat_proj], dim=1)

        ## for memory bank
        if get_image:
            return out_feat

        prompts = self.prompt_learner(img_feature_proj.detach())
        text_feat = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        sim_matrix = get_sim_matrix(img_feat_proj_patch.detach(), text_feat, [x.shape[-2], x.shape[-1]], [self.h_resolution, self.w_resolution])   
        sim_score = self.sim_with_center(sim_matrix.detach())


        # for training and evaluation
        if self.training:
            logit = self.classifier(out_feat)

            if get_matrix:
                return out_feat, logit, sim_matrix, img_feat_patch_11, sim_score
            else:
                return out_feat, logit, img_feat_patch_11, sim_score
        else:
            return out_feat, sim_score


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue # ignore classifier weights in evaluation
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_classes, camera_num, view_num):
    model = TransReID(num_classes, camera_num, view_num, cfg)
    return model

class PromptLearner(nn.Module):
    def __init__(self, dtype, token_embedding, image_feat_dim=None):
        super().__init__()
        # ctx_init = "X X X X person."
        ctx_init = "X X X X X X X X X X X X X X X X person."
        ctx_dim = 512
        n_ctx = 1  

        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # 保存 token 序列

        n_cls_ctx = 16 
        self.cls_ctx = nn.Parameter(torch.empty(n_cls_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.cls_ctx, std=0.02)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_cls_ctx:, :])
        self.n_cls_ctx = n_cls_ctx
        self.ctx_dim = ctx_dim

        if image_feat_dim is not None:
            if image_feat_dim != ctx_dim:
                self.image_feat_proj = nn.Linear(image_feat_dim, ctx_dim)
            else:
                self.image_feat_proj = nn.Identity()
        else:
            self.image_feat_proj = None

    def forward(self, label=None, image_feats_=None):
        if image_feats_ is not None:
            image_feats = image_feats_.detach()
            if self.image_feat_proj is not None:
                image_feats = self.image_feat_proj(image_feats)
            image_ctx = image_feats.unsqueeze(1)
            b = image_feats.shape[0]
            prefix = self.token_prefix.expand(b, -1, -1)
            suffix = self.token_suffix.expand(b, -1, -1)
            cls_ctx = self.cls_ctx.unsqueeze(0).expand(b, -1, -1)
            prompts = torch.cat([prefix, image_ctx+cls_ctx, suffix], dim=1)
        elif label is not None:
            b = label.shape[0]
            cls_ctx = self.cls_ctx.unsqueeze(0).expand(b, -1, -1)
            prefix = self.token_prefix.expand(b, -1, -1)
            suffix = self.token_suffix.expand(b, -1, -1)
            prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)
        else:
            cls_ctx = self.cls_ctx.unsqueeze(0)
            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)
        return prompts
