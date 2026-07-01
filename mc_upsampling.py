# -*- coding: utf-8 -*-
"""
================================================================================
 DIV2K 4x UPSAMPLING  —  o PILAR 3 do survey (Seção 4) em código
================================================================================

O QUE ESTE ARQUIVO É: um upsampler espacial 4x (renderiza/recebe uma imagem
pequena e a amplia 4x com uma rede neural). É a parte prática do pilar de
UPSAMPLING das aulas.

    O DIV2K é um dataset de FOTOS, e eu crio a imagem de baixa resolução
    REDUZINDO a foto (linha 'lr = hr.resize(...)'). Então, tecnicamente, este
    demo é SUPER-RESOLUÇÃO DE FOTO — o problema de DESBORRAR que citei na Aula 7.
    Ele NÃO é o upsampling de RENDERIZAÇÃO de verdade, que é ANTIALIASING
    (consertar serrilhado) e usa MOTION VECTORS, que uma foto não tem.
    Ou seja: este código ilustra a MECÂNICA (rede + perdas + EDSR), mas o caso
    renderizado real exigiria dados renderizados com G-buffers e motion vectors.
    (Fazer essa ressalva sozinho mostra que você entendeu a diferença
     foto-vs-render que é o coração da Aula 7.)

POR QUE A VERSÃO ANTERIOR FICAVA BORRADA (quase igual ao bicúbico) — e o conserto:

  1) PERDA (a causa nº 1 do borrão) — isto é o desafio CH2 da Aula 7 (nitidez x
     estabilidade) aparecendo na prática:
       - Antes: só L1. A L1 puxa a saída para a MÉDIA -> imagem suave/borrada,
         porque adicionar textura de alta frequência "errada" AUMENTA a L1, então
         a rede prefere jogar seguro e borrar.
       - Agora: L1 + perda de GRADIENTE (bordas) + perda PERCEPTUAL (VGG).
         Isso RECOMPENSA a rede por reconstruir textura/detalhe -> mais nítido.
      "O peso de cada perda é exatamente onde eu deslizo entre
         'estável' e 'nítido' — o trade-off do slide."

  2) ARQUITETURA (EDSR-lite) — o EDSR é um dos métodos citados no survey (o Wei
     usou EDSR):
       - Blocos residuais com 'residual scaling' (0.1) -> treino estável mesmo
         com a rede profunda (16 blocos).
       - Skip global longo (cabeça -> corpo -> +cabeça).
       - Inicialização ICNR no PixelShuffle -> evita o artefato de tabuleiro
         (checkerboard).

  3) DADOS: vários patches por imagem por época -> muito mais amostras de treino.

Rodar:  python div2k_local_v2.py
"""

import os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# EDITE AQUI SE PRECISAR
TRAIN_DIR = r"C:\Users\davir\OneDrive\Documents\Mestrado UFBA\Tópicos em Computação Visual II\deep learning MC\DIV2K_train_HR\DIV2K_train_HR"
VALID_DIR = r"C:\Users\davir\OneDrive\Documents\Mestrado UFBA\Tópicos em Computação Visual II\deep learning MC\DIV2K_valid_HR\DIV2K_valid_HR"

SCALE   = 4           # fator de ampliação (4x)
EPOCHS  = 120
PATCH   = 48          # patch LR=48 -> HR=192 (patch menor = mais amostras por imagem)
BATCH   = 16
PATCHES_PER_IMG = 16  # quantos recortes por imagem por época (multiplica os dados)
WORKERS = 2           # no Windows, se der erro de multiprocessing, use 0

# pesos da perda (é AQUI que se controla o trade-off nitidez x estabilidade)
W_GRAD       = 0.5    # peso da perda de gradiente (bordas)
W_PERCEPTUAL = 0.10   # peso da perda perceptual VGG (0 desliga)
USE_PERCEPTUAL = True # baixa pesos do VGG16 na 1a vez (precisa de internet)

# visualização
N_VIS    = 4          # quantas imagens de comparação gerar
ZOOM_BOX = 64         # tamanho do quadrado de zoom (em pixels LR)


# DATASET — POR QUÊ este é o passo mais importante de entender (o caveat mora aqui)
class DIV2KLocal(Dataset):
    def __init__(self, folder, train=True):
        self.files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if not self.files:
            raise FileNotFoundError(f"Nenhuma imagem em {folder}")
        self.train = train
        # cada imagem aparece 'mult' vezes por época, com recortes aleatórios
        # diferentes -> transforma poucas imagens em muitas amostras de treino.
        self.mult = PATCHES_PER_IMG if train else 1
        print(f"{len(self.files)} imagens HR em {os.path.basename(folder)}")

    def __len__(self):
        return len(self.files) * self.mult

    def __getitem__(self, idx):
        i = idx % len(self.files)
        try:
            hr = Image.open(self.files[i]).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        W, H = hr.size
        p = PATCH * SCALE
        if W < p or H < p:
            hr = hr.resize((p, p)); W, H = p, p
        if self.train:
            # recorte aleatório + augmentação barata (espelhos e rotação 90):
            # mais variação de treino sem precisar de mais imagens.
            x = np.random.randint(0, W - p + 1)
            y = np.random.randint(0, H - p + 1)
            hr = hr.crop((x, y, x + p, y + p))
            if random.random() < 0.5: hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5: hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5: hr = hr.transpose(Image.ROTATE_90)
        else:
            w2 = (W // SCALE) * SCALE; h2 = (H // SCALE) * SCALE
            hr = hr.crop((0, 0, w2, h2))
        
        # Criamos a imagem de baixa resolução (LR) REDUZINDO a foto de alta (HR)
        # com bicúbico. Por isso este demo é SUPER-RESOLUÇÃO DE FOTO (desborrar),
        # e NÃO o upsampling de renderização (antialiasing + motion vectors).
        lr = hr.resize((hr.size[0] // SCALE, hr.size[1] // SCALE), Image.BICUBIC)
        to_t = lambda im: torch.from_numpy(
            np.asarray(im, np.float32) / 255.).permute(2, 0, 1)
        return to_t(lr), to_t(hr)      # (entrada pequena, alvo grande)


# REDE — EDSR-lite

def icnr_(weight, scale, initializer=nn.init.kaiming_normal_):
    """Inicialização ICNR para o conv antes do PixelShuffle.
    POR QUÊ: o PixelShuffle, mal inicializado, gera o artefato de TABULEIRO
    (checkerboard) — um xadrez sutil na imagem. O ICNR inicializa os pesos de um
    jeito que já começa sem esse artefato e converge melhor."""
    out_ch, in_ch, h, w = weight.shape
    sub = out_ch // (scale ** 2)
    k = torch.zeros([sub, in_ch, h, w])
    initializer(k)
    k = k.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        weight.copy_(k)


class ResBlock(nn.Module):
    # Bloco residual: aprende só a "correção" (x + body(x)) em vez de refazer tudo.
    # res_scale=0.1 multiplica a correção por 0.1 -> impede que ela cresça demais
    # e estabiliza uma rede profunda (é o truque do EDSR para empilhar 16 blocos).
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.body(x)   # residual com escala


class Upsampler(nn.Module):
    """EDSR-lite: cabeça + N blocos residuais + upsample por PixelShuffle,
       somado ao bicúbico (skip global) para a rede só aprender o RESÍDUO."""
    def __init__(self, scale=SCALE, base=64, n_blocks=16):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, base, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(base, base, 3, padding=1)
        # up_conv produz 3*scale*scale canais; o PixelShuffle rearranja esses canais
        # em pixels, aumentando a resolução de forma APRENDIDA (não fixa).
        self.up_conv = nn.Conv2d(base, 3 * scale * scale, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        icnr_(self.up_conv.weight, scale)          # init ICNR (anti-checkerboard)

    def forward(self, x):
        # base = o bicúbico clássico (o "chute burro" da Aula 7). A rede parte
        # DAQUI e só aprende o que falta além do bicúbico (o detalhe/textura).
        base = F.interpolate(x, scale_factor=self.scale,
                             mode="bicubic", align_corners=False)
        f = self.head(x)
        f = self.body_tail(self.body(f)) + f       # skip global longo (estilo EDSR)
        # saída = bicúbico + resíduo aprendido. clamp(0,1) mantém em faixa válida.
        return (base + self.ps(self.up_conv(f))).clamp(0, 1)


# PERDAS — é aqui que se resolve o borrão (o trade-off nitidez x estabilidade)

def grad_loss(pred, hr):
    """L1 nas diferenças finitas (gradientes da IMAGEM) -> enfatiza bordas/textura.

     este 'grad_loss' é uma PERDA sobre os
    gradientes da imagem, para FORÇAR bordas nítidas. Ele NÃO é a 'renderização em
    gradiente-domínio' da Aula 6 (aquela RENDERIZAVA os gradientes para o ruído
    cancelar). São coisas diferentes que por acaso usam a palavra 'gradiente'.
    Se alguém perguntar, deixe isso claro — mostra domínio do assunto."""
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]     # diferença horizontal (pred)
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]     # diferença vertical (pred)
    dx_h = hr[:, :, :, 1:] - hr[:, :, :, :-1]         # idem no gabarito
    dy_h = hr[:, :, 1:, :] - hr[:, :, :-1, :]
    # penaliza quando as BORDAS da predição diferem das bordas do gabarito
    return F.l1_loss(dx_p, dx_h) + F.l1_loss(dy_p, dy_h)


class VGGPerceptual(nn.Module):
    """Perda perceptual usando features do VGG16 (relu3_3).
    POR QUÊ: em vez de comparar pixel a pixel (o que a L1 já faz e leva ao borrão),
    ela compara as ATIVAÇÕES de uma rede pré-treinada (VGG). Isso mede 'parecença
    aos olhos humanos' — texturas e padrões — recompensando detalhe plausível.
    (Perceptual/VGG é citada no survey, ex.: Xiao usa VGG-16, Lin usa VGG-19.)"""
    def __init__(self, dev):
        super().__init__()
        import torchvision
        try:
            w = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            vgg = torchvision.models.vgg16(weights=w).features[:16]
        except Exception:
            vgg = torchvision.models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval().to(dev)
        for p in self.vgg.parameters():
            p.requires_grad = False        # o VGG é fixo; não treinamos ele
        # normalização que o VGG espera (média/desvio do ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def forward(self, pred, hr):
        pred = (pred - self.mean) / self.std
        hr   = (hr   - self.mean) / self.std
        return F.l1_loss(self.vgg(pred), self.vgg(hr))   # distância nas features


# TREINO
def train():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("dispositivo:", dev)
    dl = DataLoader(DIV2KLocal(TRAIN_DIR, train=True), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, pin_memory=(dev == "cuda"))
    net = Upsampler().to(dev)

    perc = None
    if USE_PERCEPTUAL and W_PERCEPTUAL > 0:
        try:
            perc = VGGPerceptual(dev)
            print("perda perceptual VGG: ATIVA")
        except Exception as e:
            print("nao consegui carregar o VGG (sem internet?), seguindo sem "
                  f"perceptual. Detalhe: {e}")

    opt = torch.optim.Adam(net.parameters(), 2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        net.train(); tot = 0.
        for lr, hr in dl:
            lr, hr = lr.to(dev), hr.to(dev)
            pred = net(lr)
            # >>> A PERDA COMBINADA (o coração do conserto do borrão) <<<
            # L1 (fidelidade geral) + gradiente (bordas) + perceptual (textura).
            # Sem os dois últimos termos, a rede borra para minimizar a L1.
            loss = F.l1_loss(pred, hr) + W_GRAD * grad_loss(pred, hr)
            if perc is not None:
                loss = loss + W_PERCEPTUAL * perc(pred, hr)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoca {ep+1:3d}/{EPOCHS}  loss={tot/len(dl):.4f}")

    torch.save(net.state_dict(), "upsampler_div2k_v2.pt")
    return net, dev


# MÉTRICAS
def psnr(a, b):
    # PSNR = qualidade em decibéis (quanto MAIOR, melhor). Baseado no erro
    # quadrático médio (MSE); mais dB = menos erro.
    return 10 * np.log10(1 / max(F.mse_loss(a, b).item(), 1e-9))

try:
    from skimage.metrics import structural_similarity as _sk_ssim
    def ssim(pred, hr):
        # SSIM = similaridade ESTRUTURAL (0 a 1; quanto MAIOR, melhor). Mede se
        # padrões/estruturas batem, não só o erro pixel a pixel -> mais próximo
        # da percepção humana que o PSNR.
        p = pred.permute(1, 2, 0).numpy(); h = hr.permute(1, 2, 0).numpy()
        return _sk_ssim(h, p, channel_axis=2, data_range=1.0)
except Exception:
    def ssim(pred, hr):
        return None


# VISUALIZAÇÃO
def visualizar(net, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = VALID_DIR if os.path.isdir(VALID_DIR) else TRAIN_DIR
    ds = DIV2KLocal(folder, train=False)
    net.eval()

    # escolhe N_VIS imagens espalhadas pelo conjunto
    idxs = np.linspace(0, len(ds.files) - 1, num=min(N_VIS, len(ds.files)),
                       dtype=int)

    print("\n=== Metricas (recorte com zoom) ===")
    print(f"{'img':>4} | {'PSNR bic':>9} {'PSNR rede':>10} | "
          f"{'SSIM bic':>9} {'SSIM rede':>10}")

    for n, idx in enumerate(idxs):
        lr, hr = ds[int(idx)]
        with torch.no_grad():
            pred = net(lr[None].to(dev))[0].cpu()
        # bic = a baseline bicúbica pura, para COMPARAR contra a rede.
        bic = F.interpolate(lr[None], scale_factor=SCALE,
                            mode="bicubic", align_corners=False)[0].clamp(0, 1)

        img = lambda t: t.permute(1, 2, 0).numpy()

        # POR QUÊ procurar o recorte de MAIOR desvio-padrão: é a região com mais
        # DETALHE/textura. Numa parede lisa, rede e bicúbico empatam; a diferença
        # só aparece onde há detalhe. Então medimos e mostramos zoom ali.
        _, Hh, Ww = hr.shape
        gray = hr.mean(0).numpy()
        zb = ZOOM_BOX * SCALE
        best, by, bx = -1, 0, 0
        for _ in range(60):
            yy = np.random.randint(0, max(1, Hh - zb))
            xx = np.random.randint(0, max(1, Ww - zb))
            v = gray[yy:yy+zb, xx:xx+zb].std()
            if v > best: best, by, bx = v, yy, xx
        crop = lambda t: t[:, by:by+zb, bx:bx+zb]

        # métricas medidas NO RECORTE (onde a diferença realmente importa)
        cb, cp, ch = crop(bic), crop(pred), crop(hr)
        s_b, s_p = ssim(cb, ch), ssim(cp, ch)
        s_b_str = f"{s_b:9.3f}" if s_b is not None else f"{'-':>9}"
        s_p_str = f"{s_p:10.3f}" if s_p is not None else f"{'-':>10}"
        print(f"{n:>4} | {psnr(cb,ch):9.1f} {psnr(cp,ch):10.1f} | "
              f"{s_b_str} {s_p_str}")

        # figura: linha de cima = imagem inteira (bicúbico | rede | gabarito) com o
        # retângulo do zoom; linha de baixo = os recortes ampliados lado a lado.
        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        for a, im, ti in zip(ax[0], [bic, pred, hr],
                [f"Bicubico\nPSNR {psnr(cb,ch):.1f} dB",
                 f"Rede neural\nPSNR {psnr(cp,ch):.1f} dB",
                 "Gabarito HR (real)"]):
            a.imshow(img(im)); a.set_title(ti, fontsize=13); a.axis("off")
            a.add_patch(plt.Rectangle((bx, by), zb, zb, fill=False,
                                      edgecolor="red", linewidth=2))
        for a, im, ti in zip(ax[1], [cb, cp, ch],
                ["ZOOM - Bicubico", "ZOOM - Rede neural", "ZOOM - Gabarito"]):
            a.imshow(img(im)); a.set_title(ti, fontsize=13); a.axis("off")
        plt.tight_layout()
        out = f"comparacao_{n}.png"
        plt.savefig(out, dpi=130); plt.close(fig)
        print(f"  salvo: {out}")


if __name__ == "__main__":
    if not os.path.isdir(TRAIN_DIR):
        print("TRAIN_DIR nao existe:", TRAIN_DIR)
    else:
        net, dev = train()
        visualizar(net, dev)