# 📚 Sequence Modeling & Language Learning Timeline — *ScratchSeq (PyTorch Focused)*

This roadmap traces the evolution of **sequence modeling**, **language understanding**, and **temporal data processing** — from statistical n-gram models to transformer-based large language models — emphasizing key innovations, papers, and what to implement for hands-on learning.

---

## 1. **Foundations: Pre-Neural Sequence Models**

### 🧠 **n-Gram Models (1950s–1990s)** – ⚙️ *Must Implement*

* **Key Innovation:** Predict tokens using a fixed-length context window (Markov assumption).
* **Learning Focus:** Probability estimation (MLE), smoothing (Laplace, Good-Turing, Kneser–Ney), cross-entropy & perplexity.
* **Core Statistical References (Conceptual):**

  * Jurafsky & Martin — *Speech and Language Processing* (Language Modeling chapters)
  * Chen & Goodman (1999) — *An Empirical Study of Smoothing Techniques for Language Modeling*
* **Neural Transition Paper (Read After Implementing n-grams):**

  * **[Statistical Language Models Based on Neural Networks (Bengio, 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)**

> Bengio (2003) **assumes n-gram knowledge** and replaces discrete counts with learned embeddings. It is best read **after** implementing classical n-grams, as a bridge to neural sequence models.

---

### 🧠 **Hidden Markov Models (HMMs, 1960s–1980s)** – ⚙️📖 *Simplified Implementation*

* **Key Innovation:** Sequence modeling via latent Markov states.
* **Learning Focus:** Forward–Backward algorithm, Viterbi decoding, EM (Baum–Welch).
* **Paper:**

  * **[A Tutorial on Hidden Markov Models (Rabiner, 1989)](https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf)**

---

## 2. **Early Neural Sequence Models**

### 🧠 **Elman RNN (1990)** – ⚙️ *Must Implement*

* **Key Innovation:** Introduced recurrent hidden states for temporal memory.
* **Learning Focus:** Implement BPTT (Backprop Through Time); observe vanishing/exploding gradients.
* **Paper:**

  * **[Finding Structure in Time](https://crl.ucsd.edu/~elman/Papers/fsit.pdf)**

---

### 🧠 **Jordan Network (1990)** – 📖 *Read Only*

* **Key Innovation:** Feedback from output layer to hidden layer.
* **Learning Focus:** Contrast with Elman RNN (output feedback vs. state feedback).

---

## 3. **Gated Recurrent Units**

### 🧠 **LSTM (1997)** – ⚙️ *Must Implement*

* **Key Innovation:** Gating mechanisms to preserve long-term dependencies.
* **Learning Focus:** Build LSTM from scratch; visualize cell states; analyze gradient flow.
* **Paper:**

  * **[Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)**

---

### 🧠 **GRU (2014)** – ⚙️📖 *Partial Implementation*

* **Key Innovation:** Simplified version of LSTM with fewer gates.
* **Learning Focus:** Compare LSTM vs. GRU in speed and performance.
* **Paper:**

  * **[Learning Phrase Representations Using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078)**

---

## 4. **Sequence-to-Sequence Learning**

### 🧠 **Seq2Seq (2014)** – ⚙️ *Must Implement*

* **Key Innovation:** Encoder–decoder framework for sequence translation.
* **Learning Focus:** Build character-level Seq2Seq model; understand context vector bottlenecks.
* **Paper:**

  * **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)**

---

### 🧠 **Attention Mechanism (2015)** – ⚙️📖 *Partial Implementation*

* **Key Innovation:** Soft alignment to overcome fixed-length context bottleneck.
* **Learning Focus:** Add Bahdanau or Luong attention to Seq2Seq; visualize attention maps.
* **Paper:**

  * **[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)**

---

## 5. **The Attention Era**

### 🧠 **Transformer (2017)** – ⚙️ *Must Implement*

* **Key Innovation:** Replaced recurrence with self-attention; introduced positional encoding.
* **Learning Focus:** Multi-head attention, feedforward blocks, residual connections.
* **Paper:**

  * **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**

---

### 🧠 **BERT (2018)** – ⚙️📖 *Partial Implementation*

* **Key Innovation:** Bidirectional pretraining with Masked Language Modeling.
* **Learning Focus:** Implement masked token prediction; compare to autoregressive models.
* **Paper:**

  * **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)**

---

### 🧠 **GPT (2018–2020)** – ⚙️ *Must Implement*

* **Key Innovation:** Decoder-only autoregressive pretraining for generation.
* **Learning Focus:** Implement causal masking, next-token prediction, sampling (greedy/top-k).
* **Paper:**

  * **[Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised)**

---

## 6. **Scaling and Efficiency**

### 🧠 **Transformer-XL (2019)** – 📖 *Read Only*

* **Key Innovation:** Segment-level recurrence for long context retention.
* **Paper:**

  * **[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)**

---

### 🧠 **Reformer (2020)** – 📖 *Read Only*

* **Key Innovation:** Locality-sensitive hashing for memory-efficient attention.
* **Paper:**

  * **[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)**

---

### 🧠 **Longformer / Performer / FlashAttention (2020–2023)** – 📖 *Read Only*

* **Key Innovation:** Sparse, linear, or hardware-optimized attention.
* **Learning Focus:** Study scalable attention mechanisms and performance trade-offs.

---

## 7. **Multimodal & Instruction Models**

### 🧠 **CLIP (2021)** – ⚙️📖 *Partial Implementation*

* **Key Innovation:** Contrastive joint training of image and text embeddings.
* **Learning Focus:** Implement InfoNCE loss; align embeddings from two modalities.
* **Paper:**

  * **[CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)**

---

### 🧠 **Flamingo / BLIP-2 (2022–2023)** – 📖 *Read Only*

* **Key Innovation:** Vision-language models combining frozen LLMs with visual encoders.
* **Learning Focus:** Understand multimodal cross-attention and bridging techniques.

---

### 🧠 **Instruction-Tuned Transformers (2022–2024)** – 📖 *Read Only*

* **Key Innovation:** Fine-tuning LLMs on instruction datasets with RLHF.
* **Learning Focus:** Conceptual understanding of supervised + reinforcement alignment.

---

## 8. **Audio, Temporal, and Specialized Models**

### 🧠 **WaveNet (2016)** – ⚙️📖 *Partial Implementation*

* **Key Innovation:** Dilated causal convolutions for autoregressive waveform synthesis.
* **Learning Focus:** 1D convolutional generation; causal structure enforcement.
* **Paper:**

  * **[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)**

---

### 🧠 **Tacotron (2017)** – 📖 *Read Only*

* **Key Innovation:** End-to-end text-to-speech using attention-based sequence models.
* **Paper:**

  * **[Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)**

---

## 🧭 Implementation Path (Recommended PyTorch Order)

| Phase                        | Focus                           | What to Implement        | Key Concepts                              |
| ---------------------------- | ------------------------------- | ------------------------ | ----------------------------------------- |
| **1️⃣ Statistical Models**   | Probability-based text modeling | n-Gram, HMM              | MLE, smoothing, Markov assumptions        |
| **2️⃣ Recurrent Networks**   | Sequential dependencies         | Elman RNN, LSTM          | BPTT, vanishing gradients                 |
| **3️⃣ Seq2Seq Learning**     | Sequence mapping                | Seq2Seq + Attention      | Context vectors, alignment                |
| **4️⃣ Attention Revolution** | Parallel computation            | Transformer, GPT         | Multi-head attention, positional encoding |
| **5️⃣ Scaling**              | Efficiency & long context       | Transformer-XL, Reformer | Recurrence, memory compression            |
| **6️⃣ Multimodal Models**    | Cross-domain learning           | CLIP                     | Contrastive alignment                     |
| **7️⃣ Specialized Domains**  | Audio & temporal modeling       | WaveNet                  | Causal convolution, autoregression        |

---

## ✅ Summary

* **Must Implement:** n-Gram, Elman RNN, LSTM, Seq2Seq, Transformer, GPT-mini
* **Partial Implementation:** HMM, GRU, Attention, BERT, CLIP, WaveNet
* **Read Only:** Jordan RNN, Transformer-XL, Reformer, Longformer, Flamingo, Tacotron, Instruction-tuned models

---