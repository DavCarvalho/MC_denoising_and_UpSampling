"""
================================================================================
 DENOISER DE MONTE CARLO PATH TRACING  —  o "coração teórico" da apresentação
================================================================================

O QUE ESTE ARQUIVO É, EM UMA FRASE:
    É a Seção 3 do survey virada código. Ele implementa as DUAS arquiteturas
    centrais de denoising e, na prática, as Equações 7, 8, 9 e 10 do paper.

MAPA CÓDIGO  ->  SURVEY (decore este mapa para a apresentação):
    - Equação 7  (o treino):        classe de treino  ->  argmin_θ ℓ(c_p, g(X_p;θ))
    - X_p        (a entrada):       net_in = [ruidoso + albedo + normal + depth]
    - g          (a rede):          UNet
    - Equação 8  (Direct):          DirectPredictionDenoiser (a saída JÁ É a cor)
    - Equação 9  (softmax):         F.softmax(...) no KernelPredictionDenoiser
    - Equação 10 (média ponderada): (patches * weights).sum(...)
    - Auxiliary features (G-buffers): albedo, normal, depth  (a parte "limpa")
    - Tratamento HDR:               log1p / expm1

    "Este arquivo é a teoria dos slides rodando. Nenhuma linha é mágica nova:
     o dataset monta o X_p da Equação 7, a UNet é o g, e a diferença entre
     Direct e Kernel Prediction é literalmente a cabeça de saída da rede."

Formato dos dados (o mesmo dos papers):
    entrada   = render ruidoso em baixo spp (ex.: 4 spp)  -> 3 canais RGB
    G-buffers = albedo (3) + normal (3) + depth (1)        -> 7 canais auxiliares
    gabarito  = render limpo em alto spp (ex.: 8192 spp)   -> o alvo c_p
    => cada exemplo: tensor [10, H, W] de entrada  e  [3, H, W] de alvo.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# SEÇÃO 1 · DATASET  —  aqui montamos o X_p (entrada) e o c_p (alvo) da Equação 7
#
# POR QUÊ este passo existe: denoising é APRENDIZADO SUPERVISIONADO. Precisamos
# de pares (imagem ruidosa + features -> imagem limpa de referência) para a rede
# aprender comparando. Este dataset entrega exatamente esses pares.
class MCRenderDataset(Dataset):
    """
    Espera uma pasta com pares .npz. Cada cena_XXX.npz contém:
        noisy  : float32 [H, W, 3]   render em baixo spp (radiância HDR)  -> parte do X_p
        albedo : float32 [H, W, 3]   G-buffer (a "cor base" do material)  -> parte do X_p
        normal : float32 [H, W, 3]   G-buffer (para onde a superfície aponta)
        depth  : float32 [H, W, 1]   G-buffer (distância à câmera)
        clean  : float32 [H, W, 3]   referência em alto spp                -> o alvo c_p

    >>> NA HORA (sobre os G-buffers):
        "Esses 3 canais extras (albedo, normal, depth) são as 'auxiliary
         features' do slide. Elas vêm LIMPAS da geometria (não têm ruído de
         Monte Carlo) e dizem à rede ONDE NÃO BORRAR: se dois pixels vizinhos
         têm normais bem diferentes, são objetos diferentes -> não misture ->
         a borda é preservada."

    Truques padrão dos papers embutidos aqui:
      - log-transform na radiância HDR (log(1+x)) para amansar outliers/fireflies
      - recorte de patches aleatórios (ex.: 128x128) -> mais amostras de treino
    """

    def __init__(self, root_dir, patch_size=128, train=True):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"Nenhum .npz em {root_dir}")
        self.patch = patch_size
        self.train = train

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _log_tonemap(x):
        # POR QUÊ log1p: radiância é HDR — um pixel de céu pode valer 10000 e um
        # de sombra 0,01. Sem comprimir, os pixels brilhantes dominam a perda e a
        # rede ignora as sombras. log(1+x) "esmaga" essa faixa gigante e reduz o
        # peso dos fireflies (pontos brancos espúrios do MC).
        # >>> NA HORA: "É o mesmo problema das perdas HDR que citei no denoising."
        return torch.log1p(torch.clamp(x, min=0.0))

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        # permute(2,0,1): converte [H,W,C] (formato de imagem) para [C,H,W] (formato
        # que o PyTorch/convoluções esperam). Só reorganiza os eixos, não muda dado.
        noisy  = torch.from_numpy(d["noisy"]).permute(2, 0, 1).float()
        albedo = torch.from_numpy(d["albedo"]).permute(2, 0, 1).float()
        normal = torch.from_numpy(d["normal"]).permute(2, 0, 1).float()
        depth  = torch.from_numpy(d["depth"]).permute(2, 0, 1).float()
        clean  = torch.from_numpy(d["clean"]).permute(2, 0, 1).float()

        if self.train:
            # POR QUÊ recortar patches: uma imagem 1080p é grande demais para a GPU
            # e daria pouquíssimos exemplos. Recortando pedaços aleatórios, a mesma
            # imagem vira MUITAS amostras de treino diferentes a cada época.
            _, H, W = noisy.shape
            ps = self.patch
            y = torch.randint(0, max(H - ps, 1), (1,)).item()
            x = torch.randint(0, max(W - ps, 1), (1,)).item()
            sl = (slice(None), slice(y, y + ps), slice(x, x + ps))
            noisy, albedo, normal, depth, clean = (
                t[sl] for t in (noisy, albedo, normal, depth, clean)
            )

        # Aplica o log na ENTRADA e no ALVO (os dois têm que estar no mesmo espaço).
        noisy_log = self._log_tonemap(noisy)
        clean_log = self._log_tonemap(clean)

        # AQUI nasce o X_p da Equação 7: empilhamos ruidoso(3) + albedo(3) +
        # normal(3) + depth(1) = 10 canais. É o "bloco de dados" que a rede vê.
        net_in = torch.cat([noisy_log, albedo, normal, depth], dim=0)
        return net_in, noisy_log, clean_log


# SEÇÃO 2 · BACKBONE U-NET  —  esta é a função g(·;θ) da Equação 7
#
# POR QUÊ U-Net: é o formato "funil" (encoder comprime -> gargalo -> decoder
# reconstrói) com ATALHOS (skip connections). O encoder captura o contexto amplo
# (bom para juntar muitos vizinhos e tirar ruído); os atalhos trazem de volta os
# detalhes finos que a compressão perderia. O survey aponta a U-Net/autoencoder
# como a espinha dorsal de quase todos os denoisers.
# >>> NA HORA: "Este é o 'g' genérico. As DUAS variantes usam a MESMA U-Net;
#              só muda a cabeça de saída — e é aí que está a diferença dos slides."
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net pequeno, no estilo dos denoisers em tempo real do survey."""

    def __init__(self, in_ch=10, out_ch=3, base=48):
        super().__init__()
        # ENCODER: cada nível dobra os canais e (via pool) reduz a resolução pela
        # metade -> a rede "enxerga cada vez mais longe" ao redor de cada pixel.
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(base * 4, base * 8)      # gargalo (visão mais global)
        # DECODER: sobe a resolução de volta. O torch.cat no forward é o ATALHO que
        # devolve os detalhes finos do encoder para o decoder.
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)          # camada final (o "z_L" do paper)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        # cada 'cat' junta a versão upsampled com a feature original do encoder (o atalho)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# SEÇÃO 3a · DIRECT PREDICTION  —  a Equação 8 do survey: c_hat = z_L
#
# IDEIA: a saída da rede JÁ É a cor limpa. Sem filtro, sem restrição.
# É o lado LARANJA do slide "Direct vs Kernel": mais poderoso (pode inventar
# detalhe), mas menos estável e pode gerar cores inválidas.
# É a abordagem que OptiX e OIDN (indústria) usam.
class DirectPredictionDenoiser(nn.Module):
    def __init__(self, in_ch=10):
        super().__init__()
        self.net = UNet(in_ch=in_ch, out_ch=3)

    def forward(self, net_in, noisy_log):
        # DETALHE PRÁTICO: em vez de a rede refazer a cor do zero, ela prevê a
        # CORREÇÃO (o resíduo) e nós somamos à imagem ruidosa. Isso converge mais
        # rápido, porque a rede só precisa aprender "o conserto", não a imagem toda.
        # >>> NA HORA (se perguntarem): "Continua sendo Direct Prediction: a saída
        #     é a cor, sem a restrição de convex hull do kernel."
        return noisy_log + self.net(net_in)


# SEÇÃO 3b · KERNEL PREDICTION / KPCN  —  as Equações 9 e 10 (Bako et al. 2017)
#
# IDEIA (o lado TEAL do slide): a rede NÃO cospe a cor. Ela prevê os PESOS de um
# filtro, e a cor final é a MÉDIA PONDERADA dos vizinhos ruidosos. Como os pesos
# são positivos e somam 1 (softmax), a saída fica no "convex hull" dos vizinhos:
# a rede só mistura cores que já existem, nunca inventa valor inválido.
# CONSEQUÊNCIA: treino mais estável e rápido (o preço é ficar preso à janela k×k).
class KernelPredictionDenoiser(nn.Module):
    """
    A rede produz k*k logits por pixel. A softmax (Eq. 9) vira pesos w_pq >= 0 que
    somam 1. A cor denoised (Eq. 10) é a soma ponderada da vizinhança k x k.
    """

    def __init__(self, in_ch=10, kernel_size=21):
        super().__init__()
        self.k = kernel_size
        # A cabeça de saída tem k*k canais: um PESO para cada vizinho da janela k×k.
        # (compare com a Direct, que tinha out_ch=3 — a cor. Aqui a saída são pesos.)
        self.net = UNet(in_ch=in_ch, out_ch=kernel_size * kernel_size)

    def forward(self, net_in, noisy_log):
        B, _, H, W = noisy_log.shape
        k = self.k

        logits = self.net(net_in)                    # pesos CRUS (podem ser qualquer nº)
        # >>> ESTA LINHA É A EQUAÇÃO 9 <<<
        # A softmax transforma os pesos crus em pesos válidos: todos positivos e
        # somando 1. É o que GARANTE o convex hull e estabiliza o treino.
        weights = F.softmax(logits, dim=1)

        # F.unfold extrai a vizinhança k×k de CADA pixel da imagem ruidosa.
        # É a "janela N(p)" do slide: os vizinhos que serão misturados.
        pad = k // 2
        patches = F.unfold(noisy_log, kernel_size=k, padding=pad)
        patches = patches.view(B, 3, k * k, H, W)    # [B, 3(RGB), k*k(vizinhos), H, W]

        # >>> ESTA LINHA É A EQUAÇÃO 10 <<<
        # c_hat = Σ_q  c_q * w_pq   (soma ponderada dos vizinhos).
        # weights.unsqueeze(1): os MESMOS pesos são aplicados aos 3 canais RGB —
        # é o detalhe do slide de que "o kernel é compartilhado pelos canais de cor".
        out = (patches * weights.unsqueeze(1)).sum(dim=2)  # [B, 3, H, W]
        return out


# SEÇÃO 4 · LOSS  —  o ℓ(·,·) da Equação 7
#
# L1 = erro absoluto médio. É o "ℓ" que mede a diferença entre a saída da rede e a
# referência. Simples e comum. Nos papers (Vogels, NPPD) usa-se muito o SMAPE (erro
# relativo, melhor para HDR) — no arquivo de dados reais nós trocamos por SMAPE.
def denoising_loss(pred, target):
    return F.l1_loss(pred, target)


# SEÇÃO 5 · TREINO  —  o argmin_θ da Equação 7, feito por gradiente descendente
def train(data_dir, arch="kpcn", epochs=100, batch_size=8, lr=1e-4,
          device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = MCRenderDataset(data_dir, patch_size=128, train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)

    # Aqui escolhemos QUAL cabeça de saída usar (a mesma U-Net por dentro).
    if arch == "kpcn":
        model = KernelPredictionDenoiser(in_ch=10, kernel_size=21)
    else:
        model = DirectPredictionDenoiser(in_ch=10)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # CosineAnnealingLR: reduz o learning rate suavemente até o fim -> ajuste fino
    # nas épocas finais (boa parte do ganho de qualidade vem no finalzinho).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for net_in, noisy_log, clean_log in dl:
            net_in = net_in.to(device)
            noisy_log = noisy_log.to(device)
            clean_log = clean_log.to(device)

            pred = model(net_in, noisy_log)          # g(X_p; θ)
            loss = denoising_loss(pred, clean_log)   # ℓ(c_p, g(X_p; θ))

            opt.zero_grad()
            loss.backward()
            # clip_grad_norm: "corta" gradientes grandes demais. POR QUÊ importa
            # aqui: predição direta pode ter gradientes que explodem por causa de um
            # firefly (pixel HDR gigante); isso é exatamente o ponto de instabilidade
            # do slide. O clip segura o treino para não desestabilizar.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()                               # o passo do argmin_θ
            total += loss.item()

        sched.step()
        print(f"epoch {epoch+1:3d}/{epochs}  loss={total/len(dl):.5f}")

    torch.save(model.state_dict(), f"denoiser_{arch}.pt")
    print(f"Modelo salvo em denoiser_{arch}.pt")
    return model


# SEÇÃO 6 · INFERÊNCIA  —  aplica o modelo treinado numa imagem inteira
@torch.no_grad()
def denoise_image(model, npz_path, device="cpu"):
    model.eval().to(device)
    d = np.load(npz_path)
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1).float()[None]
    noisy = to_t(d["noisy"]).to(device)
    aux = torch.cat([to_t(d["albedo"]), to_t(d["normal"]),
                     to_t(d["depth"])], dim=1).to(device)

    noisy_log = torch.log1p(torch.clamp(noisy, min=0.0))
    net_in = torch.cat([noisy_log, aux], dim=1)

    pred_log = model(net_in, noisy_log)
    # expm1 DESFAZ o log1p: voltamos do "espaço-log" para a radiância HDR real.
    # >>> NA HORA: "Treinamos no espaço-log; aqui desfazemos para recuperar o HDR."
    pred = torch.expm1(pred_log).clamp(min=0.0)
    return pred[0].permute(1, 2, 0).cpu().numpy()    # [H, W, 3] HDR


# SEÇÃO 7 · DADOS SINTÉTICOS  —  para testar o pipeline sem um renderizador
def make_fake_dataset(out_dir, n=16, H=256, W=256):
    """Simula renders: gabarito suave + ruído ~ Poisson (parecido com o MC real)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
        clean = np.stack([
            0.5 + 0.5 * np.sin(6 * xx + rng.uniform(0, 6)),
            0.5 + 0.5 * np.cos(5 * yy + rng.uniform(0, 6)),
            0.4 + 0.4 * np.sin(4 * (xx + yy)),
        ], axis=-1).astype(np.float32)
        spp = 4
        # POR QUÊ ruído de POISSON (e não Gaussiano): o ruído de Monte Carlo nasce
        # de CONTAR eventos aleatórios (raios/fótons que chegam), e contagem de
        # eventos raros segue a distribuição de Poisson. Então isto imita o ruído
        # MC de forma mais fiel que um ruído gaussiano genérico.
        # >>> NA HORA: bom detalhe para mostrar que você entende a ORIGEM do ruído.
        noisy = rng.poisson(clean * spp * 20).astype(np.float32) / (spp * 20)
        albedo = clean.copy()
        normal = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float32)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        depth = (xx * yy)[..., None].astype(np.float32)
        np.savez(os.path.join(out_dir, f"cena_{i:03d}.npz"),
                 noisy=noisy, albedo=albedo, normal=normal,
                 depth=depth, clean=clean)
    print(f"{n} exemplos sintéticos salvos em {out_dir}")


if __name__ == "__main__":
    # Teste rápido de ponta a ponta com dados falsos (não precisa de renderizador):
    make_fake_dataset("data_fake", n=16)
    model = train("data_fake", arch="kpcn", epochs=5, batch_size=4)
    result = denoise_image(model, "data_fake/cena_000.npz")
    print("Imagem denoised:", result.shape, result.dtype)