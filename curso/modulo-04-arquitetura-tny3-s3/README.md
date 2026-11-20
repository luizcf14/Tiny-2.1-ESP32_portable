# Módulo 4 — Arquitetura Moderna em 16 MB (`TNY3-S3`): RoPE, GQA, QAT e Flash `mmap`

> **Carga horária sugerida:** 5 horas  
> **Hardware padrão:** `ESP32-S3-WROOM-1 (16MB)` (`N16`, `N16R2` e `N16R8`)  
> **Arquivos de referência do repo:** [`TinyESP32/partitions.csv`](../../TinyESP32/partitions.csv), [`TinyESP32/tiny_engine.cpp`](../../TinyESP32/tiny_engine.cpp), [`tools/finetune_tiny.py`](../../tools/finetune_tiny.py), [`tools/generate_esp32_assets.py`](../../tools/generate_esp32_assets.py)

---

## Objetivos de Aprendizagem

Ao concluir este módulo, o aluno será capaz de:
1. Projetar a arquitetura **`TNY3-S3`** (~7,1 milhões de parâmetros) usando técnicas modernas de LLMs (RoPE, GQA e SwiGLU) escolhidas especificamente para economizar RAM e Flash.
2. Comparar quantização **`int8`** (por linha) versus **`int4` (`Q4_0` em blocos de 32 pesos)** e aplicar **QAT (*Quantization-Aware Training*)** em PyTorch.
3. Criar uma **tabela de partições de 16 MB** ([`partitions_s3_16mb.csv`](../../TinyESP32/partitions.csv) como referência de evolução) com partição neural dedicada de **8 MB** mapeada em Zero-Copy via **`esp_partition_mmap()`**.
4. Dimensionar o **KV Cache** para rodar mesmo em placas sem PSRAM (`N16`) e escalar automaticamente quando houver PSRAM (`N16R2`/`N16R8`).

---

## 1. De `TNY2` (2024/DOS) para `TNY3-S3` (Arquitetura Moderna no ESP32-S3)

Nos últimos anos, a evolução de arquiteturas abertas (Llama, Gemma, Qwen) trouxe inovações cujo principal motivo de existir é **reduzir gargalos de memória e largura de banda**. No ESP32-S3, esses mesmos truques são ainda mais decisivos:

```mermaid
flowchart LR
    TOK["Token ID 0..4095"] --> EMB["emb 4096x256 -> bx"]
    EMB --> NORM1["RMSNorm"]
    NORM1 --> QKV["Wq 8 heads | Wk 2 heads | Wv 2 heads"]
    QKV --> ROPE["RoPE: rotação cos/sin em Q e K - sem tabela pos"]
    ROPE --> GQA["GQA: 4 Q heads compartilham 1 KV head"]
    GQA --> RESID1["Wo + Residual"]
    RESID1 --> NORM2["RMSNorm"]
    NORM2 --> SWI["SwiGLU: SiLU Gate x Up -> Down"]
    SWI --> RESID2["Residual -> Próxima camada 8x"]
```

### Os 3 Pilares Arquiteturais do `TNY3-S3`

#### Pilar 1 — RoPE (*Rotary Positional Embeddings*) no lugar da tabela `pos`
* No `TNY2`, existe uma tabela fixa `pos[CTX][D]` aprendida na Flash, que trava o modelo no tamanho exato de `CTX = 128`.
* Com **RoPE**, a posição `p` entra multiplicando pares de coordenadas `(x[2i], x[2i+1])` dos vetores `Q` e `K` por `cos(p * theta_i)` e `sin(p * theta_i)`.
* **Benefício embarcado:** 0 parâmetros extras de tabela posicional para aprender; basta uma pequena tabela pré-calculada de `cos/sin` (`CTX × HD`, ~32 KB para `CTX=512, HD=32`) ou cálculo direto por recorrência trigonométrica.

#### Pilar 2 — GQA (*Grouped-Query Attention*) para Salvar a SRAM
* No `TNY2` (`MHA`), temos `H = 4` cabeças de Query e `4` cabeças de Key/Value.
* No **`TNY3-S3`** com `D = 256`, usamos **`H_Q = 8` cabeças de Query (`HD = 32`)**, mas apenas **`H_KV = 2` cabeças de Key/Value** (cada grupo de 4 cabeças `Q` lê o mesmo par `K, V`).
* **Impacto imediato no KV Cache:** O consumo cai **4x**! Veja os números exatos para `L = 8` camadas, `H_KV = 2` (`KV_DIM = 64`), com quantização dinâmica `int8` (`64 B + 4 B escala = 68 B` por vetor):

| Janela (`CTX`) | KV Cache MHA Clássico (`8 KV heads`) | KV Cache `TNY3-S3` com **GQA (`2 KV heads`)** | Onde Roda no ESP32-S3-WROOM-1 (16MB)? |
| :---: | :---: | :---: | :--- |
| **`256` tokens** | `~1,04 MB` *(não cabe sem PSRAM!)* | **`278.528 B` (~272 KB)** | Cabe direto na **SRAM interna** (`N16`, `N16R2` e `N16R8`)! |
| **`512` tokens** | `~2,08 MB` | **`557.056 B` (~544 KB)** | **PSRAM** (`N16R2` / `N16R8`) via `tiny_alloc_kv()` |
| **`1024` tokens** | `~4,16 MB` | **`1.114.112 B` (~1,06 MB)** | **PSRAM** (`N16R2` / `N16R8`) |

> **Lição de ouro:** Sem GQA, um modelo de 8 camadas com `D=256` nem ligaria na variante `N16` (sem PSRAM). Com GQA `8:2`, ele cabe com folga na SRAM para `CTX = 256` (~200 palavras) ou usa `CTX = 512+` automaticamente se detectar PSRAM!

#### Pilar 3 — SwiGLU no bloco FFN
* Em vez de `f2(ReLU(f1(x)))`, usamos `W_down(SiLU(W_gate(x)) * W_up(x))`.
* Para manter o mesmo orçamento de parâmetros de uma FFN tradicional com expansão `4×`, o tamanho intermediário é ajustado para `~2,68 × D` (`HID = 688`, múltiplo de 16 para alinhamento SIMD!).
* *Alternativa didática:* O aluno também pode comparar no mesmo esqueleto **ReLU²** (`max(0, x)^2`), que exige apenas 2 matrizes em vez de 3 e entrega qualidade excelente em modelos de < 10M parâmetros.

---

## 2. Orçamento Exato de Parâmetros e Quantização (`int8` vs `Q4_0`)

Vamos somar os pesos da configuração recomendada do curso (**`TNY3-S3`: `VOCAB=4096, D=256, L=8, H_Q=8, H_KV=2, HID=688`, Weight Tying na saída**):

| Bloco | Dimensões | Parâmetros | Tamanho em `int8` (TNY2-style) | Tamanho em `int4` (`Q4_0` bloco 32) |
| :--- | :--- | :---: | :---: | :---: |
| **Embeddings (`emb` tied)** | `4096 × 256` | `1.048.576` | `1,02 MB` | `0,56 MB` *(recomendado manter `emb` em `int8`)* |
| **Atenção (`Wq, Wk, Wv, Wo` × 8)** | `8 × (256² + 64×256 + 64×256 + 256²)` | `1.310.720` | `1,27 MB` | `0,70 MB` |
| **FFN SwiGLU (`Wg, Wu, Wd` × 8)** | `8 × 3 × (688 × 256)` | `4.227.072` | `4,10 MB` | `2,27 MB` |
| **Norms (`g1, g2, gf` em `float32`)** | `(8×2 + 1) × 256` | `4.352` | `17 KB` | `17 KB` |
| **Total do Modelo `TNY3-S3`** | — | **~6,59M params** | **~6,41 MB** | **~3,55 MB** *(ou ~4,0 MB híbrido)* |

### Entendendo o Formato `Q4_0` (Blocos de 32 Pesos em 4 Bits)

Como o gargalo em C é ler bytes da Flash SPI, reduzir um peso de `8 bits` para `4 bits` **corta pela metade o tráfego de leitura na Flash**:
* Cada bloco de **32 pesos consecutivos** armazena: `1 escala float16/float32` + `16 bytes` (onde cada byte guarda **2 pesos de 4 bits com sinal `[-8, +7]`**, um no nibble baixo e outro no nibble alto).
* Com QAT (`ste_int4`) adaptado diretamente do `ste_int8` de [`tools/finetune_tiny.py`](../../tools/finetune_tiny.py#L119-L123), a perda de qualidade em relação a `int8` fica mínima enquanto a pegada de Flash cai quase 45%!

---

## 3. Saindo de `tiny_weights.h` para Partição de 16 MB + `esp_partition_mmap()`

No [`TinyESP32/tiny_weights.h`](../../TinyESP32/tiny_weights.h) atual (`381 KB`), os pesos são compilados dentro do binário do programa como um array gigante em C. Se você fizer isso com **6,4 MB** de pesos:
1. O compilador `g++` demora minutos para compilar um header de ~20 MB de texto hexadecimal.
2. O arquivo resultante excede o limite de partição de aplicação padrão (`app0`).
3. Toda vez que você treina uma época nova, precisa recompilar e regravar todo o código C++!

### A Solução Profissional: Tabela `partitions_s3_16mb.csv` (16 MB)

Substituímos a lógica do [`TinyESP32/partitions.csv`](../../TinyESP32/partitions.csv) (que visava placas de 2 MB+) por um mapa sob medida para os **16 MB (0x1000000 bytes)** do `ESP32-S3-WROOM-1`:

```csv
# Name,      Type, SubType, Offset,    Size,      Flags
nvs,         data, nvs,     0x9000,    0x5000,
otadata,     data, ota,     0xE000,    0x2000,
app0,        app,  ota_0,   0x10000,   0x280000,
model_bin,   data, 0x99,    0x290000,  0x800000,
rag_kb,      data, 0x9A,    0xA90000,  0x400000,
spiffs,      data, spiffs,  0xE90000,  0x160000,
coredump,    data, coredump,0xFF0000,  0x10000,
```

| Partição | Offset | Tamanho | Papel no Curso |
| :--- | :--- | :---: | :--- |
| `app0` | `0x10000` | **2,5 MB** | Firmware `TinyESP32` + Web UI + Stack Wi-Fi/BLE (compila em segundos!) |
| `model_bin` | `0x290000` | **8,0 MB** | Arquivo binário `TNY3` gravado direto com `esptool write_flash 0x290000 model.bin` |
| `rag_kb` | `0xA90000` | **4,0 MB** | Base de conhecimento indexada offline (Módulo 5 — Micro-RAG) |
| `spiffs` | `0xE90000` | **1,37 MB** | Logs, histórico persistente ou configurações extras |

### Como Ler 8 MB da Flash com 0 Bytes de RAM (`esp_partition_mmap`)

No código C++ do ESP32-S3 (`#include <esp_partition.h>`):

```c
const esp_partition_t *part = esp_partition_find_first(
    ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x99, "model_bin");

const void *mapped_ptr = NULL;
esp_partition_mmap_handle_t mmap_handle;
esp_err_t err = esp_partition_mmap(
    part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &mmap_handle);
```

Depois dessas 5 linhas, `mapped_ptr` vira um ponteiro normal `const uint8_t *` passado exatamente como antes para `tiny_init(mapped_ptr, model_bytes, &err)`! A **Flash MMU do ESP32-S3** mapeia páginas de 64 KB sob demanda para o cache de dados do processador.

---

## 4. Laboratório Prático — Validando o Layout `TNY3` e Gravando em Partição

1. **Estudo do Novo Cabeçalho `TNY3` (32 bytes alinhados):**
   Projetar o cabeçalho estendido suportando:
   * `magic`: `"TNY3"`
   * `version`, `quant_mode` (`1 = int8`, `2 = Q4_0`)
   * `vocab_size` (`4096`), `ctx_len` (`256`/`512`), `d` (`256`), `layers` (`8`), `q_heads` (`8`), `kv_heads` (`2`), `hid` (`688`)
   * `rope_theta` (`10000.0f`), `vocab_offset`, `weights_offset`.
2. **Simulador Zero-Copy em Python:**
   Estender a lógica do [`tools/generate_esp32_assets.py`](../../tools/generate_esp32_assets.py) para validar alinhamento de `TNY3` e gerar um binário dummy inicial, testando o mapeamento `esp_partition_mmap()` no ESP32-S3 real (verificando o ponteiro e o header `"TNY3"` lido da Flash no Serial Monitor).
3. **Upload Flash Independente:**
   Gravar a partição `model_bin` sem recompilar o firmware usando `esptool.py --chip esp32s3 write_flash 0x290000 models/tny3_model.bin` e cronometrar o tempo de troca de modelo.

---

## 5. Exercícios do Módulo

1. **Orçamento na `N16` vs `N16R8`:** Considerando a fórmula do KV Cache com GQA (`L × CTX × 2 × (H_KV × HD + 4)`), calcule o `CTX` máximo seguro que ainda deixa ao menos 120 KB de heap livre na SRAM para o Wi-Fi em uma placa `N16` sem PSRAM.
2. **Por que manter `emb` em `int8` mesmo com FFN em `Q4_0`?** Analise a sensibilidade de pesos compartilhados entre entrada e classificação final de 4.096 classes frente ao erro de quantização de 4 bits.
3. **Verificação da MMU:** No ESP32-S3, quantos megabytes de espaço de endereço virtual DROM podem ser mapeados simultaneamente pela MMU? Consulte o datasheet técnico do ESP32-S3 para confirmar por que uma partição de 8 MB cabe dentro da janela de `esp_partition_mmap`.

---

| Anterior | Índice | Próximo |
| :--- | :---: | :--- |
| [← Módulo 3: Tokenizador BPE Embarcado](../modulo-03-tokenizador-bpe-embarcado/README.md) | [Guia do Curso](../README.md) | [Módulo 5: Fábrica de Dados e Destilação →](../modulo-05-fabrica-dados-destilacao/README.md) |
