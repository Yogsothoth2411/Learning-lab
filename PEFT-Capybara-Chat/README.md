# PEFT-Capybara-Chat

本專案為 **LLM 微調實驗 (Parameter-Efficient Fine-Tuning, PEFT)**，使用 `trl-lib/Capybara` 對話資料集，嘗試以 **QLoRA（LoRA + 4-bit 量化）** 技術對 `Llama-3.1-8B-Instruct` 模型進行低成本微調。


## 專案簡介

此專案的目標在於：

* 熟悉 `QLoRA` 微調流程 與 `Hugging Face PEFT` 工具鏈。
* 嘗試 **低資源環境（RTX 3050, 8GB VRAM）** 進行 4-bit 量化訓練的可行性與穩定性。
* 觀察微調後模型在語意生成上的表現（以 BERTScore 衡量）。

## 使用模型與資料集

| 項目               | 名稱 / 來源                                                             |
| ------------------ | ---------------------------------------------------------------------- |
| Base Model         | `meta-llama/Llama-3.1-8B-Instruct`                                     |
| Dataset            | [`trl-lib/Capybara`](https://huggingface.co/datasets/trl-lib/Capybara) |
| Fine-tuning Method | PEFT (QLoRA)                                                           |
| Quantization       | BitsAndBytes 4-bit (`nf4`, `float16` compute dtype)                    |

## 實驗架構簡述

1. **資料處理**
   * 從 `Capybara` dataset 取出多輪對話結構。
   * 擷取 `user` 與 `assistant` 訊息。
   * 分為 `train` (500 筆) 與 `test` (200 筆)。

2. **模型載入**
   * 使用 `transformers` 與 `BitsAndBytesConfig` 載入 4-bit 量化模型。
   * 設定 `tokenizer.pad_token = eos_token` 與 `device_map="auto"`。

3. **訓練設定**
   * 採用 `PEFT` + `QLoRA`。
   * `batch_size=2`，搭配梯度累積以避免 GPU OOM。
   * 訓練完成後保存 adapter 權重。

4. **效果評估**
   * 使用 `evaluate` 套件中的 `bertscore`。
   * 以生成結果評估 fine-tuned model 的語意相似度。


## 實驗結果

| 模型版本          | Precision | Recall | F1         |
| ----------------- | --------- | ------ | ---------- |
| Fine-tuned (PEFT) | 0.7891    | 0.8190 | **0.8031** |

> 此結果為使用 QLoRA 在 8GB VRAM 下完成的微調成果，
> 展示了低資源環境下進行大型模型參數高效訓練的可行性。

## 執行環境

| 組件                | 說明                                                               |
| ----------------- | ---------------------------------------------------------------- |
| GPU               | RTX 3050 (8GB VRAM)                                              |
| Frameworks        | PyTorch, Hugging Face Transformers, PEFT, BitsAndBytes, Evaluate |
| Dataset Framework | Hugging Face Datasets                                            |
| OS                | Windows 11                                                       |


## 參考資源

* [Hugging Face PEFT 官方文件](https://huggingface.co/docs/peft/index)
* [TRL: Training Transformer Language Models](https://huggingface.co/docs/trl)
* [Capybara Dataset](https://huggingface.co/datasets/trl-lib/Capybara)

