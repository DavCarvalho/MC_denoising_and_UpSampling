# -*- coding: utf-8 -*-
"""
================================================================================
 TREINO DO KPCN COM DADOS REAIS DO NOISEBASE  —  a teoria aplicada de verdade
================================================================================

O QUE ESTE ARQUIVO É: pega o KernelPredictionDenoiser do mc_denoiser.py (as
Eqs. 9-10) e o treina em dados REAIS de path tracing, o NOISEBASE.

>>> NA HORA (abertura deste arquivo):
    "Aqui o mesmo denoiser dos slides roda em dados de verdade. E ele usa DOIS
     conceitos específicos que dei em aula: a DEMODULAÇÃO POR ALBEDO (Chaitanya)
     e a perda SMAPE (Vogels/NPPD)."

DE ONDE VÊM OS DADOS (conexão bonita para citar):
    O NOISEBASE é o dataset do NPPD (Balint et al.) — que é justamente o método
    de SAMPLE SPACE que citei na aula de kernel prediction. Ou seja, estou
    treinando com dados de um dos próprios papers do survey.

O QUE ELE FAZ:
    1) baixa algumas cenas do NOISEBASE (uma vez, sem apagar),
    2) treina o KernelPredictionDenoiser nelas,
    3) valida em cenas separadas e salva o melhor modelo,
    4) gera uma figura de comparação [ RUIDOSO | DENOISED | REFERENCIA ] por cena.

DUAS DECISÕES QUE VALEM OURO PARA A BANCA (explicadas nos comentários abaixo):
  [1] LOSS = SMAPE (erro relativo) em vez de L1.
      POR QUÊ: a L1/L2 no espaço-log fica "perseguindo" fireflies (pixels HDR
      muito brilhantes) e empaca o treino. O SMAPE é RELATIVO -> um erro de 10%
      conta igual na sombra e no céu -> equilibra claro/escuro. É o que os papers
      (Vogels, NPPD) usam. (Isto conecta com o slide de perdas HDR do denoising.)

  [2] VALIDAÇÃO HONESTA -> cenas [6, 11] (mesma distribuição do treino).
      POR QUÊ: a validação antiga usava cenas [1023, 1022], que davam loss ~100x
      menor que o treino. Isso NÃO era "generaliza ótimo": eram cenas atípicas
      (escuras/lisas) onde qualquer método ganha fácil, e a escolha do "melhor
      modelo" virava quase aleatória. A cena 6 já ficou DE FORA do treino (veja
      CENAS), então é uma validação honesta e "in-distribution".
      >>> NA HORA: "É a diferença entre uma métrica que engana e uma que informa."

DEMODULAÇÃO — a observação honesta para a dissertação:
    Usamos DEMODULAÇÃO POR ALBEDO (dividir a radiância pelo albedo antes de
    filtrar e multiplicar de volta depois). É a técnica de Chaitanya et al./NPPD
    (Sec. 3.1.1 do survey). NÃO é o "split difuso/especular" do Bako — esse
    exigiria as radiâncias difusa e especular separadas, que o NOISEBASE não dá.
    >>> NA HORA: citar esta distinção mostra que você entende a diferença entre
        os dois truques (Chaitanya vs Bako).

PRE-REQUISITOS (rode no terminal, uma vez):
    pip install torch noisebase pillow "zarr<3"
    #                                   ^^^^^^^^ o noisebase quebra com zarr 3.x
    #                                   (erro 'zarr has no ZipStore').

DISCO: cada cena tem ~2 a 9 GB. Com 10 cenas, conte ~30-90 GB.
PASTA: deixe FORA do OneDrive (evita travas e sincronização gigante).
"""

import os
import numpy as np
import torch
from urllib.parse import urljoin

from mc_denoiser import KernelPredictionDenoiser
from noisebase.loaders.torch.training_sample_v1 import TrainingSampleDataset
from noisebase.torch import Shuffler


# CONSTANTES — edite aqui (e só aqui)
PASTA       = "D:/nb_data"     # onde salvar os zips (FORA do OneDrive!)
CENAS       = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]   # cenas de TREINO (note: 6 fica de fora)
CENAS_TESTE = [6, 11]          # validação HONESTA, mesma distribuição do treino
EPOCHS      = 40
BATCH       = 2                # reduza p/ 1 se faltar memória/VRAM
SAMPLES     = 8                # spp do input ruidoso (<=32). Baixo spp = mais ruído.
KERNEL      = 21               # tamanho k da janela do kernel; reduza p/ 15 se faltar VRAM
LR          = 1e-4
DEMODULAR   = True             # liga/desliga a demodulação por albedo (Chaitanya)
EPS         = 1e-2             # pequeno número para evitar divisão por ~0
SAIDA_BASE  = "comparacao_denoiser"
FRAMES_FIG  = [0, 20, 40]


# Config interna do dataset (valores oficiais do sampleset_v1 do NOISEBASE)
REMOTE = "https://neural-partitioning-pyramids.mpi-inf.mpg.de/data/"
SRC = {
    "files": "sampleset_v1/train/scene{index:04d}.zip",
    "frames_per_sequence": 64, "crop": 256, "samples": 32,
    "rendering_height": 1080, "rendering_width": 1920, "sequences": 1024,
}
# Os buffers que pedimos ao dataset. 'diffuse' aqui faz o papel de ALBEDO.
# color=ruidoso, reference=gabarito limpo, normal/depth=G-buffers.
BUFFERS = ["color", "diffuse", "normal", "depth", "reference"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# LOSS: SMAPE no espaço-log — o "ℓ" da Equação 7, mas na versão ROBUSTA a HDR.
#
# SMAPE = Symmetric Mean Absolute Percentage Error (erro percentual simétrico).
# POR QUÊ é melhor que L1 aqui: ele DIVIDE o erro pela magnitude dos pixels, então
# vira um erro RELATIVO. Assim ele não é dominado pelos pixels brilhantes (não
# "persegue" fireflies) e trata sombra e céu com o mesmo peso.
def denoising_loss(pred, target):
    num = (pred - target).abs()
    den = pred.abs() + target.abs() + EPS   # o +EPS evita dividir por ~0 nas sombras
    return (2.0 * num / den).mean()


# 1) Download de uma cena (pula se já existe no disco)
def baixar_cena(cena):
    import requests
    rel = SRC["files"].format(index=cena)
    local = os.path.join(PASTA, rel)
    if os.path.exists(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    temp = local + ".part"                   # baixa num arquivo temporário
    print(f"  baixando cena {cena} ...")
    with requests.get(urljoin(REMOTE, rel), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(temp, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(temp, local)                  # só renomeia no fim -> download atômico
    return local


# 2) Ler frames de uma cena e montar um batch
def dataset_da_cena(cena, flip_rotate):
    # flip_rotate=True liga a AUGMENTAÇÃO (espelhar/rotacionar) — só no treino,
    # para a rede ver mais variação sem precisar de mais dados.
    return TrainingSampleDataset(
        src=SRC, sequence_idxs=[cena], rng=Shuffler(42),
        flip_rotate=flip_rotate, buffers=BUFFERS, samples=SAMPLES,
        data_path=PASTA, batch_size=1,
    )


def media_amostras(t):
    # O NOISEBASE entrega as amostras individuais [B,C,H,W,S] (S = nº de amostras).
    # Aqui fazemos a MÉDIA sobre S -> colapsa para [B,C,H,W].
    # >>> NA HORA (conceito importante): "Fazer essa média me coloca em PIXEL SPACE
    #     (o mais barato dos 3 espaços). Se eu quisesse SAMPLE SPACE — melhor para
    #     outliers — eu NÃO faria a média e teria que tratar a invariância à
    #     permutação, que é bem mais complexo."
    t = torch.as_tensor(t).float()
    if t.dim() == 5:        # [B,C,H,W,S] -> [B,C,H,W]
        t = t.mean(dim=-1)
    return t


def monta_batch(ds, indices_de_frame, epoch):
    acc = {k: [] for k in BUFFERS}
    for f in indices_de_frame:
        fr = ds[{"idx": f, "epoch": epoch}]
        for k in BUFFERS:
            acc[k].append(torch.as_tensor(np.ascontiguousarray(fr[k])))
    return {k: torch.stack(v, 0) for k, v in acc.items()}


def prepara(batch):
    # Extrai cada buffer e faz a média das amostras (-> pixel space).
    noisy  = media_amostras(batch["color"]).to(DEVICE)
    albedo = media_amostras(batch["diffuse"]).to(DEVICE)   # 'diffuse' = albedo
    normal = media_amostras(batch["normal"]).to(DEVICE)
    depth  = media_amostras(batch["depth"]).to(DEVICE)
    clean  = media_amostras(batch["reference"]).to(DEVICE)
    if depth.shape[1] != 1:
        depth = depth[:, :1]
    # Normaliza a profundidade para [0,1] (cada cena tem escala diferente; sem isso
    # a rede veria números com magnitudes muito diferentes entre cenas).
    dmin = depth.amin(dim=(1, 2, 3), keepdim=True)
    dmax = depth.amax(dim=(1, 2, 3), keepdim=True)
    depth = (depth - dmin) / (dmax - dmin + 1e-6)

    # >>> DEMODULAÇÃO POR ALBEDO (o truque de Chaitanya — slide de denoising) <<<
    # A cor ruidosa mistura ILUMINAÇÃO (ruidosa) com TEXTURA/albedo (nítida e já
    # conhecida). Dividindo pelo albedo, tiramos a textura e deixamos só a
    # iluminação suave — que é MUITO mais fácil de limpar. Depois (no 'remodular')
    # multiplicamos o albedo de volta para recuperar a textura nítida.
    if DEMODULAR:
        noisy = noisy / (albedo + EPS)
        clean = clean / (albedo + EPS)

    # log HDR (igual ao mc_denoiser): comprime a faixa dinâmica.
    noisy_log = torch.log1p(noisy.clamp(min=0))
    clean_log = torch.log1p(clean.clamp(min=0))
    # X_p da Equação 7: ruidoso(3) + albedo(3) + normal(3) + depth(1) = 10 canais.
    net_in = torch.cat([noisy_log, albedo, normal, depth], dim=1)
    return net_in, noisy_log, clean_log, albedo


def remodular(pred_log, albedo):
    # Desfaz o log (expm1) e, se demodulamos, MULTIPLICA o albedo de volta ->
    # devolve a textura nítida que tínhamos tirado. É a metade final do truque.
    img = torch.expm1(pred_log).clamp(min=0)
    return img * (albedo + EPS) if DEMODULAR else img


# 3) Validação (SMAPE médio nas cenas de teste) -> escolhe o melhor modelo
@torch.no_grad()
def validar(model):
    model.eval()
    total, n = 0.0, 0
    nframes = SRC["frames_per_sequence"]
    for cena in CENAS_TESTE:
        ds = dataset_da_cena(cena, flip_rotate=False)   # sem augmentação na validação
        for i in range(0, nframes, BATCH):
            batch = monta_batch(ds, range(i, min(i + BATCH, nframes)), epoch=0)
            net_in, noisy_log, clean_log, _ = prepara(batch)
            total += denoising_loss(model(net_in, noisy_log), clean_log).item(); n += 1
    model.train()
    return total / max(n, 1)


# 4) Figuras de comparação: uma por cena de teste [ RUIDOSO | DENOISED | REFERENCIA ]
def tonemap(x, gamma=2.2):
    # Converte HDR (faixa enorme) para uma imagem de 8 bits exibível na tela.
    # x/(1+x) comprime o brilho (tone mapping de Reinhard); a potência 1/gamma
    # faz a correção gama (como o monitor espera).
    x = np.clip(x, 0, None); x = x / (1 + x); x = np.power(x, 1 / gamma)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def salvar_figura(model, cena):
    from PIL import Image, ImageDraw
    model.eval()
    ds = dataset_da_cena(cena, flip_rotate=False)

    linhas = []
    for frame in FRAMES_FIG:
        batch = monta_batch(ds, [frame], epoch=0)
        net_in, noisy_log, clean_log, albedo = prepara(batch)
        pred_log = model(net_in, noisy_log)
        denoised = remodular(pred_log, albedo)          # remodula (albedo de volta)
        noisy    = remodular(noisy_log, albedo)         # o ruidoso, também remodulado
        ref      = media_amostras(batch["reference"]).to(DEVICE)
        np_img = lambda t: t[0].permute(1, 2, 0).cpu().numpy()
        # cola as 3 imagens lado a lado: ruidoso | denoised | referência
        linha = np.concatenate(
            [tonemap(np_img(noisy)), tonemap(np_img(denoised)), tonemap(np_img(ref))],
            axis=1)
        linhas.append(linha)

    grade = np.concatenate(linhas, axis=0)              # empilha os frames escolhidos
    W3 = grade.shape[1]
    faixa = 28
    canvas = np.full((grade.shape[0] + faixa, W3, 3), 255, np.uint8)
    canvas[faixa:] = grade
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    col = W3 // 3
    for i, txt in enumerate(["RUIDOSO", "DENOISED", "REFERENCIA"]):
        d.text((i * col + col // 2 - 35, 7), txt, fill=(0, 0, 0))
    nome = f"{SAIDA_BASE}_cena{cena}.png"
    img.save(nome)
    print(f"figura salva: {nome}  ({len(FRAMES_FIG)} frames x 3 colunas)")


# 5) PROGRAMA PRINCIPAL — o laço de treino (o argmin_θ da Eq. 7 em dados reais)
def main():
    print(f"device = {DEVICE} | treino = {CENAS} | teste = {CENAS_TESTE} | "
          f"demodular = {DEMODULAR} | loss = SMAPE")
    if DEVICE == "cpu":
        print("AVISO: sem GPU, o KPCN fica lento. Considere Colab/Kaggle/cluster.")
    os.makedirs(PASTA, exist_ok=True)

    for c in CENAS + CENAS_TESTE:
        baixar_cena(c)

    # O MESMO KernelPredictionDenoiser do mc_denoiser.py (Eqs. 9-10).
    model = KernelPredictionDenoiser(in_ch=10, kernel_size=KERNEL).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    nframes = SRC["frames_per_sequence"]
    melhor_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        soma, npassos = 0.0, 0
        for c in CENAS:
            ds = dataset_da_cena(c, flip_rotate=True)   # augmentação ligada no treino
            ordem = list(range(nframes)); np.random.shuffle(ordem)
            for i in range(0, nframes, BATCH):
                idxs = ordem[i:i + BATCH]
                batch = monta_batch(ds, idxs, epoch=epoch)
                net_in, noisy_log, clean_log, _ = prepara(batch)

                pred = model(net_in, noisy_log)          # g(X_p; θ)
                loss = denoising_loss(pred, clean_log)   # SMAPE(c_p, g(...))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # estabilidade
                opt.step()
                soma += loss.item(); npassos += 1
        sched.step()

        val = validar(model)
        # treino e teste na MESMA métrica (SMAPE) -> os dois números são
        # diretamente comparáveis (foi uma das correções metodológicas).
        print(f"epoch {epoch+1:3d}/{EPOCHS}  treino(SMAPE)={soma/max(npassos,1):.5f}  "
              f"teste(SMAPE)={val:.5f}")
        # salva o modelo SÓ quando a validação melhora (early-model selection):
        # evita ficar com um modelo que decorou o treino mas piorou no teste.
        if val < melhor_val:
            melhor_val = val
            torch.save(model.state_dict(), "kpcn_melhor.pt")
            print("   -> melhor modelo salvo: kpcn_melhor.pt")

    model.load_state_dict(torch.load("kpcn_melhor.pt", map_location=DEVICE))
    for cena in CENAS_TESTE:
        salvar_figura(model, cena)
    print("\npronto.")


if __name__ == "__main__":
    main()