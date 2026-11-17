# TINY 2.1 — Porte para ESP32 (Arduino IDE)

Este diretório adiciona o porte completo do **TINY 2.1** (originalmente escrito em Turbo C 2.0 para MS-DOS) para o **ESP32 clássico (WROOM / DevKit V1, 4 MB Flash, sem PSRAM)**, mantendo intactos todos os arquivos originais de DOS (`TINY.C`, `TINY.BIN`, `TINY.EXE`, `BUILD.BAT`, `README.TXT`).

## O que foi adaptado para o ESP32

- **Zero-Copy na Flash (0 bytes de RAM para os pesos):** Os 381.808 bytes de `TINY.BIN` (`367.000` parâmetros, 4 camadas, 96 dimensões, 4 cabeças, FFN 256) ficam armazenados em Flash (`PROGMEM` / `DROM`) via [`TinyESP32/tiny_weights.h`](TinyESP32/tiny_weights.h), com leitura alinhada de `float` + `int8_t` diretamente pela MMU da Flash.
- **KV Cache no Heap:** Os buffers de contexto (`kc`, `vc`, `ks`, `vs` — cerca de 100 KB) são alocados via `malloc()` durante `tiny_init()` antes de iniciar o Wi-Fi, evitando estouro do segmento `.bss` (`dram0_0_seg`) no Xtensa e deixando bastante heap livre para a pilha TCP/IP.
- **FPU Single-Precision & Watchdog:** Toda a matemática usa `float` de 32 bits (`sqrtf`, `expf`), desenrolamento 4x em `matvec` e `yield()` por token para não disparar o Task Watchdog Timer.
- **Zero bibliotecas externas:** Usa apenas bibliotecas nativas do core oficial `esp32` da Espressif (`WiFi`, `WebServer`, `Preferences`, `ESPmDNS`).
- **Duas interfaces completas:**
  1. **Serial USB (115200 baud):** Perguntas e respostas em streaming direto no Serial Monitor da Arduino IDE (com `/help`, `/demo`, `/0`..`/9`, `/status`, `/wifi`).
  2. **Interface Web Neon via Wi-Fi:** Réplica fiel da interface VGA neon do DOS (chuva digital de boot, scanner de leitura, `NEURAL MONITOR` ao vivo com `LAYER ACTIVITY`, `CONFIDENCE`, `LAST LETTERS` e `SPEED` via Server-Sent Events, além de voz robótica PC-Speaker via Web Audio API).

---

## Como compilar e gravar na Arduino IDE

1. Certifique-se de ter o pacote de placas **esp32 by Espressif Systems** instalado no *Boards Manager* (compatível com as versões **2.x** e **3.x**).
2. Abra o sketch [`TinyESP32/TinyESP32.ino`](TinyESP32/TinyESP32.ino) na **Arduino IDE**.
3. No menu **Tools (Ferramentas)**, selecione:
   - **Board:** `ESP32 Dev Module` (ou a placa ESP32 WROOM equivalente de 4 MB)
   - **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)` (para placas de 4 MB+)
     *(A pasta já inclui [`TinyESP32/partitions.csv`](TinyESP32/partitions.csv) com `app0` de `0x1E0000` / ~1,9 MB e fim em `0x200000`, funcionando automaticamente em placas de **2 MB, 4 MB, 8 MB e 16 MB**.)*
   - **Upload Speed:** `921600` (ou `115200` se houver falha na gravação)
4. Clique em **Upload** e aguarde a gravação.

---

## Como usar

### 1. Pelo Serial Monitor
- Abra o **Serial Monitor** em `115200 baud` (ajuste o fim de linha para `Newline` ou `Both NL & CR`).
- Digite uma pergunta em inglês (o modelo foi treinado em inglês com ~3.500 fatos) e pressione **Enter**:
  - `what is the capital of peru`
  - `is a whale a fish`
  - `what is 7 times 8`
  - `who invented the telephone`
  - `spell elephant`
- Comandos especiais no Serial:
  - `/0` .. `/9`: ajusta a criatividade (`0` = resposta mais exata; `5` = padrão)
  - `/demo`: executa as perguntas de demonstração automaticamente
  - `/status`: mostra dimensões da rede, uso de RAM/Flash e IPs Wi-Fi
  - `/wifi <ssid> <senha>`: conecta o ESP32 ao seu roteador Wi-Fi e salva na NVS
  - `/wifi clear`: apaga as credenciais do roteador
  - `/help`: lista os comandos

### 2. Pela Interface Web Neon (Wi-Fi)
- O ESP32 cria automaticamente o Access Point aberto **`TINY-2.1-ESP32`**.
- Conecte seu celular ou computador a essa rede Wi-Fi e abra no navegador:
  - **`http://192.168.4.1`**
- Opcionalmente, clique no botão **`WIFI`** na barra inferior da interface web (ou use `/wifi` no Serial) para conectar o ESP32 ao roteador da sua casa/escritório; após conectar, também fica acessível por **`http://tiny.local`** (mDNS).
- Dica de áudio: clique ou pressione qualquer tecla ao carregar a página para liberar o áudio do navegador e ouvir os efeitos PC-Speaker + voz robótica.

---

## Consumo Estimado de Memória (ESP32 Clássico 4 MB)

| Recurso | Local | Tamanho | Observação |
| --- | --- | --- | --- |
| Pesos `TINY.BIN` | Flash (`PROGMEM`/DROM) | **381.808 B** (~373 KB) | Zero-copy direto da Flash (0 B de RAM) |
| Interface Web (`index.html.gz`) | Flash (`PROGMEM`/DROM) | **~9,3 KB** | Servida com `Content-Encoding: gzip` |
| KV Cache (`kc`, `vc`, `ks`, `vs`) | SRAM Heap (`malloc`) | **102.400 B** (100 KB) | Alocado no `setup()` antes do Wi-Fi |
| Buffers do Transformer (`bx..bs`) | SRAM Estática | **~5,5 KB** | Reutilizados a cada token |
| Firmware + Pilha Wi-Fi / WebServer | Flash (`app0` 1,87 MB) | **~1,2–1,4 MB** total | Cabe com folga em `partitions.csv` (2 MB+) |

---

## Solução de Problemas (ESP32-S3 / ESP32-C3 / USB CDC On Boot)

Se você fez upload e **não apareceu nada no Serial Monitor**:

1. **USB CDC On Boot (`Tools -> USB CDC On Boot: Enabled`):**
   - Em placas com USB nativo (**ESP32-S3, ESP32-C3, ESP32-C6, ESP32-S2**), o cabo USB demora **2–4 segundos** para reenumerar a porta COM no Windows após o upload/Reset.
   - O sketch agora aguarda a enumeração CDC, espelha saída simultânea em `Serial` + `Serial0` (útil em placas **ESP32-S3** com duas portas `USB` e `UART`), reenvia o status quando o Serial Monitor é aberto depois e suporta envio mesmo em modo `"No Line Ending"`.
2. **Ajustes recomendados no menu `Tools` se a rede Wi-Fi `TINY-2.1-ESP32` não aparecer:**
   - **ESP32-S3 (ex.: DevKitC-1 N16R8):** `USB Mode: Hardware CDC and JTAG`, `USB CDC On Boot: Enabled`, `Flash Mode: DIO` (ou `QIO 80MHz` se suportado), `PSRAM: OPI PSRAM` (em placas R8).
   - **ESP32-C3 SuperMini / DevKit:** `USB CDC On Boot: Enabled`, `Flash Mode: DIO`.
   - Depois de gravar, pressione o botão **RST (Reset)** na placa uma vez com o Serial Monitor aberto em `115200 baud`.

---

## Como fazer Fine-Tuning do modelo (`tools/finetune_tiny.py`)

O repositório inclui o script [`tools/finetune_tiny.py`](tools/finetune_tiny.py) com **PyTorch + QAT (Quantization-Aware Training em int8)** para ajustar o `TINY.BIN` com seus próprios fatos, idioma (português/inglês) ou personalidade, exportando no formato binário exato `TNY2` do ESP32.

1. **Edite seu dataset de perguntas e respostas:**
   - Formato simples texto em [`data/custom_qa.txt`](data/custom_qa.txt) (`pergunta | resposta`):
     ```txt
     qual e o seu nome | Meu nome e TINY 2.1.
     onde voce esta rodando | Estou rodando direto em um ESP32 sem PSRAM.
     ```
   - Ou formato JSONL em [`data/custom_qa.jsonl`](data/custom_qa.jsonl) com múltiplas formulações para a mesma resposta:
     ```json
     {"q": ["qual a capital da italia", "capital da italia"], "a": "A capital da Italia e Roma."}
     ```
   - Acentos em UTF-8 (`ã`, `ç`, `é`, `ó`...) são convertidos automaticamente para ASCII tanto no treino quanto na entrada do usuário no ESP32 (`tiny_clean_question`).

2. **Treine e atualize o sketch do ESP32 em 1 comando:**
   ```bash
   python tools/finetune_tiny.py --data data/custom_qa.txt --update-esp32
   ```
   Isso treina sobre `TINY.BIN` (sem sobrescrever o arquivo original), salva em `models/tiny_finetuned.bin`, testa as respostas já quantizadas em `int8` e atualiza [`TinyESP32/tiny_weights.h`](TinyESP32/tiny_weights.h). Depois basta clicar em **Upload** na Arduino IDE.

3. **Teste no PC antes de gravar no ESP32:**
   ```bash
   python tools/finetune_tiny.py --chat --model models/tiny_finetuned.bin
   ```

4. **Opções úteis de ajuste:**
   - `--epochs 40` / `--lr 4e-4`: aumente se houver muitos exemplos novos ou se o modelo ainda errar alguma resposta; reduza para ajustar com mais suavidade.
   - `--replay-ratio 0.25`: mistura fatos originais do `TINY.BIN` durante o treino para evitar esquecimento catastrófico (use `0` para focar 100% no seu dataset).
   - `--l2-sp 0.005`: mantém os pesos ancorados ao modelo base.
   - `--no-augment`: desliga variações automáticas com/sem `?` e caixa baixa.
   - `--restore-esp32`: restaura [`TinyESP32/tiny_weights.h`](TinyESP32/tiny_weights.h) para o `TINY.BIN` original de fábrica.

---

## Regerar os assets caso edite `web/index.html` ou troque `TINY.BIN`

```bash
python tools/generate_esp32_assets.py
python tools/verify_port.py
```
