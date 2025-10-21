
# LangChain 檢索器與 RAG 評估實作

本專案展示了使用公開資料集（SQuAD 2.0）和 LangChain 框架檢索增強生成 (Retrieval-Augmented Generation, RAG) 系統，並詳細展示了**檢索器 (Retriever)** 的構建、優化，以及**端到端評估流程**的實作。

---

## 1. 專案說明

此專案主要目標是**建立並評估 RAG 系統中的檢索環節效能**，特別著重於比較不同檢索策略的表現。

使用了 **SQuAD 2.0 官方版本數據集**（僅保留可回答問題，即 `is_impossible=False`）進行資料載入與抽樣測試。

專案涵蓋了從數據載入、前處理、向量索引構建、檢索器組合，到最終使用多種指標進行評估的完整流程。

## 2. 核心功能

1.  **數據準備與轉換**：
    *   載入並抽樣 **SQuAD 2.0 數據集**（初步測試使用了 4000 筆資料）。
    *   將資料轉換為 LangChain 的 `Document` 格式和 QA 配對格式（用於評估）。

2.  **檢索器構建與優化**：
    *   實現稀疏檢索器 **BM25**。
    *   實現稠密檢索器 **FAISS** (基於 HuggingFace 嵌入模型構建)。
    *   構建**混合檢索器 (Ensemble Retriever)**，結合 BM25 和 FAISS 的結果，權重可調整 (例如：`[0.2, 0.8]`)。
    *   整合 **Cross-Encoder Reranker** (交叉比對重排序器) 作為上下文壓縮器，用於優化檢索結果的排名。

3.  **評估指標與方法**：
    *   **檢索器指標 (Retriever Evaluation)**：自定義函數計算 **Recall@k**、**Precision@k** 和 **MRR** (Mean Reciprocal Rank)。
    *   **生成答案指標 (Answer Evaluation)**：使用 LangChain 的 `QAEvalChain` 進行 **LLM 輔助評估**，以及計算 **BERTScore (F1 Mean)** 來評估生成答案的準確性。
    > 備註：BLEU / ROUGE 在 SQuAD 2.0 短文本生成的情況下效果有限，因此本專案主要使用 BERTScore 作為語義匹配指標。

## 3. 模型引用

本專案在檢索、重排序和生成階段使用了以下模型：

| 組件 | 模型名稱 | 引用來源 |
| :--- | :--- | :--- |
| **大型語言模型 (LLM)** | `meta-llama/Llama-3.1-8B-Instruct` | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| **嵌入模型 (Embedding)** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | [HuggingFace](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| **交叉比對重排序器 (Reranker)** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | [HuggingFace](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |

>本專案使用之模型皆依 HuggingFace 提供之原始授權使用。

> **LLM 載入說明**：Llama-3.1-8B 模型採用 4-bit 量化配置（BitsAndBytesConfig）進行載入，以優化 GPU 記憶體使用。

## 4. 批量限制說明

在**階段 5：Langchain evaluation 測試**中，進行批量生成和評估時遇到限制：

*   **評估樣本限制**：為了控制測試複雜度和運行時間，在 LangChain 評估環節中，僅使用了前 150 筆 QA 配對進行測試。

## 5. 執行方式

專案執行流程分為五個主要階段，以下為程式碼的邏輯步驟：

### 階段 1：資料載入與前處理
1.  載入 SQuAD 2.0 數據集 (`squad_v2`)。
2.  定義 Pydantic 數據模型 `testdata`。
3.  從訓練集抽樣 4000 筆資料。
4.  將抽樣資料轉換為用於檢索的 LangChain `Document` 列表 (`train_docs`) 和用於評估的 QA 配對列表 (`qa_pairs`)。

### 階段 2：文本切分與嵌入 (FAISS index 構建)
1.  創建 `RecursiveCharacterTextSplitter`，設定 `chunk_size=300` 和 `chunk_overlap=100`。
2.  使用 `HuggingFaceEmbeddings` 載入嵌入模型。
3.  構建 **FAISS** 向量庫，並將切分後的文件加入其中。

### 階段 3：Retriever 創建與問答鏈構建
1.  初始化 **BM25Retriever**（使用 `word_tokenize` 進行英文分詞處理）。
2.  創建 **FAISSRetriever**（`k=10`）。
3.  構建 **EnsembleRetriever** (混合檢索)，設定權重 (`weights=[0.2, 0.8]`)。
4.  初始化 **CrossEncoderReranker** (使用 `cross-encoder/ms-marco-MiniLM-L-6-v2`)。
5.  創建 **ContextualCompressionRetriever**，將混合檢索器與重排序器組合，設定重排序後取前 3 名 (`top_n=3`)。
6.  載入 Llama 3.1 LLM 模型並創建 **RetrievalQA 問答鏈**，使用壓縮檢索器作為輸入。

### 階段 4：測試 Recall、Precision、MRR
1.  執行自定義的 `evaluate_retriever` 函數，使用壓縮檢索器 (`compression_retriever`) 和 QA 配對數據。
2.  計算並輸出 **Recall@k**、**Precision@k** 和 **MRR** (設定 $k=3$)。

### 階段 5：Langchain evaluation 測試 BERT Score
1.  將 QA 配對數據轉換為 HuggingFace Dataset 格式（僅使用前 100 筆）。
2.  定義 `generate_and_eval` 函數，循環調用 `qa_chain.invoke` 生成預測，並使用 **`QAEvalChain`** 進行評估。
3.  計算 **BERTScore (F1 mean)** 來評估有答案情況下的生成答案品質。

## 6. 檢索器與評估流程

### 檢索器配置

| 檢索類型 | 組件 | 關鍵參數 |
| :--- | :--- | :--- |
| **稀疏檢索器** | BM25Retriever | `k = 10` |
| **稠密檢索器** | FAISSRetriever | `search_kwargs={"k":10}` |
| **混合檢索器** | EnsembleRetriever | `weights=[0.2, 0.8]` |
| **優化/壓縮** | CrossEncoderReranker | `top_n=3` |

### 檢索器評估結果（$k=3$）

| 指標 | 結果 | 說明 |
| :--- | :--- | :--- |
| **Recall@k** | **0.9102** | 檢索到的前 K 筆文件中包含正確答案的比例。 |
| **Precision@k** | **0.3869** | 檢索到的前 K 筆文件中相關文件的比例。 |
| **MRR** | **0.8612** | 第一個正確答案排名倒數的平均值。 |
| **評估樣本數** | 2605 | |

### 生成答案評估結果


| 指標 | 結果 | 說明 |
| :--- | :--- | :--- |
| **BERTScore (F1 mean)** | **0.8209** | 衡量生成答案與黃金標準答案之間的語義相似度與事實重合度。 |

在生成答案評估中，所有樣本均為「可回答問題」，不包含 SQuAD 2.0 中的空白（unanswerable）問題。
若需完整評估模型拒答能力，可於後續加入「no answer」型提示模板。

## 模型授權聲明
本專案僅引用並調用以下開源模型，不包含模型權重：
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) — 使用 Meta Llama 3 Community License。
- [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — 使用 Apache 2.0 License。
- [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) — 使用 Apache 2.0 License。

本專案僅作為研究與教學使用，未重新分發上述模型權重。
