# PEFT-OPT350M-Capybara-Chat

本專案為 **LLM 有監督微調實驗 (Supervised Fine-Tuning, SFT) 與 PEFT**，使用 `trl-lib/Capybara` 對話資料集，對 `facebook/opt-350m` 模型進行 **SFT + LoRA 微調**（本次未使用 4-bit 量化）。

## 專案簡介

此專案的目標在於：

* 熟悉 `SFT`（有監督式微調） 與 `PEFT` / `LoRA` 微調流程。
* 嘗試 **低資源環境（RTX 3050, 8GB VRAM）** 進行模型微調。
* 觀察微調後模型在對話生成與下游任務上的表現。


## 使用模型與資料集

| 項目                 | 名稱 / 來源                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| Base Model         | `facebook/opt-350m`                                                    |
| Dataset            | [`trl-lib/Capybara`](https://huggingface.co/datasets/trl-lib/Capybara) |
| Fine-tuning Method | PEFT (LoRA)                                                            |
| Quantization       | None (dtype=torch.float16)                                             |


## 實驗架構簡述

1. **資料處理**

   * 從 `Capybara` dataset 取出多輪對話結構。
   * 擷取 `user` 與 `assistant` 訊息。
   * 過濾過長樣本，最終 train: 3683 筆、validation: 921 筆、test: 60 筆。

2. **模型載入**

   * 使用 `transformers` 載入 `facebook/opt-350m`。
   * 設定 `tokenizer.pad_token = eos_token` 與 `device_map="auto"`。

3. **訓練設定**

   * 採用 `PEFT` + `LoRA`。
   * batch size = 2，搭配梯度累積以避免 GPU OOM。
   * 訓練完成後保存 LoRA adapter 權重。

4. **效果評估**

   * 使用 `evaluate` 套件計算 Precision / Recall / F1。
   * 評估 fine-tuned model 在對話任務上的表現。


## 實驗結果

| 模型版本    | Precision | Recall | F1     |
| ---------- | --------- | ------ | ------ |
| Baseline   | 0.8064    | 0.8360 | 0.8191 |
| SFT + LoRA | 0.8033    | 0.8350 | 0.8171 |

> SFT + LoRA 微調後與 baseline 相比，表現略下降，但在低資源環境下仍維持穩定效果。


## 執行環境

| 組件                | 說明                                                 |
| ----------------- | -------------------------------------------------- |
| GPU               | RTX 3050 (8GB VRAM)                                |
| Frameworks        | PyTorch, Hugging Face Transformers, PEFT, Evaluate |
| Dataset Framework | Hugging Face Datasets                              |
| OS                | Windows 11                                         |


## 參考資源

* [Hugging Face PEFT 官方文件](https://huggingface.co/docs/peft/index)
* [TRL: Training Transformer Language Models](https://huggingface.co/docs/trl)
* [Capybara Dataset](https://huggingface.co/datasets/trl-lib/Capybara)
* [2255 SFT-v2](https://github.com/Heng-xiu/AUO-2255-post-training)
