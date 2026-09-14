# ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding

<p align="center">
  <a href="https://github.com/YiyangZhou">Yiyang Zhou*</a>, 
  <a href="https://github.com/codepassionor">Yangfan He*</a>, 
  <a href="https://github.com/InfiniteLoopCoder">Yaofeng Su</a>, 
  <a href="https://github.com/Lillianwei-h">Siwei Han</a>, 
  <a href="https://github.com/joeljang">Joel Jang</a>, 
  <a href="https://www.gedasbertasius.com/">Gedas Bertasius</a>, 
  <a href="https://www.cs.unc.edu/~mbansal/">Mohit Bansal</a>, 
  <a href="https://www.huaxiuyao.io/">Huaxiu Yao</a>
</p>


<div align="center">
*Equal Contribution
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2506.01300"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>


**ReAgent-V** is a **modular**, **extensible**, and **reward-aware** video reasoning framework designed to elevate video question answering and reasoning through:

- 🔧 **Flexible Tool Integration** — Plug-and-play support for OCR, ASR, object detection, scene graph generation, captioning, and more  
- 🧠 **Reward-Guided Inference** — Enables real-time self-correction via structured reward signals  
- 🎯 **Adaptive Model Alignment** — Aligns models dynamically based on inference-time feedback  
- 🗂️ **High-Quality Data Selection** — Facilitates sample-efficient learning using reflective evaluation  
- 📊 **Entropy-Calibrated Frame Selection** — Prioritizes key frames for focused reasoning  
- 🔁 **Multi-Perspective Reflection** — Refines answers through debate among conservative, neutral, and aggressive viewpoints

  
## News
🚀 **Update [Sept 19, 2025]:** Our paper (arXiv:2506.01300) has been accepted to **NeurIPS 2025**! 

🔥 **\[June 2, 2025]** Our latest paper is now live on arXiv: [arXiv:2506.01300](https://arxiv.org/abs/2506.01300)!


---

# 📌 Overview
![Framework Overview](assets/framework.png)  


## 🚀 Applications

ReAgent-V supports a range of real-world tasks via dedicated application modules:

### 🧭 VLA Alignment  
Aligns **Vision-Language-Action (VLA)** models using **Trajectory-wise Preference Optimization (TPO)** guided by ReAgent-V’s reward feedback. Specifically, ReAgent-V evaluates each trajectory across multiple axes—such as task success, temporal stability, visual grounding, and semantic precision—and performs multi-agent reflection to produce re- 📁 Module: `Application/VLA-Alignment`  
- 📘 Instructions: [VLA Alignment README](Application/VLA-Alignment/README.md)

### 🎥 Video Understanding 

- **Entropy-Calibrated Frame Selection**  
  Efficiently selects the most informative frames for video reasoning.

- **Tool-Augmented Inference**  
  Dynamically integrates multimodal tools including OCR, ASR, object detection, scene graph generation, and captioning, etc.

- **Multi-Agent Reflection**  
  Iteratively refines outputs by encouraging disagreement and consensus among diverse agent personas (conservative / neutral / aggressive).

- 📁 Module: `ReAgent-V`  
- 📘 Instructions: [Video Understanding README](ReAgent-V/README.md)

### 📈 Reward-Aware Data Curation and Collection for SFT, DPO, GRPO, and Beyond

ReAgent-V enables **inference-time data curation** by leveraging real-time rewards and reflection-based diagnostics depending on the optimization paradigm, the extraction strategy varies:


#### 🧪 For **SFT (Supervised Fine-Tuning)**

ReAgent-V can directly **collect samples with high reward scores** (from the evaluation report) without requiring additional reflection.

* ✅ These samples indicate that the model’s initial reasoning is reliable.
* 📥 Stored as supervised training pairs with accompanying scalar reward labels from the critic agent.

> Simple, scalable, and label-efficient: reward scores enable dynamic filtering without manual annotation.


#### 🔄 For **GRPO (Group Relative Policy Optimization)**

To curate high-value training data for GRPO, ReAgent-V employs a **reflection-triggering mechanism** grounded in **importance scoring**, effectively identifying **challenging yet informative** video-text samples during the *video understanding phase*.

* 🎥 Each input is a **(video, text) pair**, typically comprising a video and its initial response.
* 📊 During inference, ReAgent-V computes an **importance score** (denoted as `E.importance_score`) based on the critic agent’s **overall assessment of reasoning sufficiency**.
* ❗ If this importance score falls **below a threshold** (e.g., `< 5 out of 10`), the sample is considered **difficult**, meaning the model struggled with initial reasoning and likely required further refinement.
* 📥 The resulting **(video, text)** samples are labeled as **reflection-worthy** and collected as **valuable candidates** for GRPO training.


#### ⚖️ For **DPO (Direct Preference Optimization)**

ReAgent-V supports **Direct Preference Optimization (DPO)** by reframing itself from a video reasoning agent into a **reward-generating agent**. This is achieved through a **task template modification** that emphasizes **evaluating answer quality** rather than producing a single correct answer.

* 🧠 Transforms the task from “answer the video question” into **“score the video from different perspectives”** based on visual evidence.

* ♻️ Uses multi-perspective reflection outputs (e.g., **conservative**, **neutral**, **aggressive**) to generate **candidate rewards**.

* 📊 Each candidate reward is generated along **customizable reward dimensions**, such as:

  * 🎯 Visual alignment
  * ⏱️ Temporal accuracy
  * 💬 Linguistic precision
  * 🧠 Reasoning specificity
  * 🔍 Option disambiguation

* ✅ The system identifies the answer with the **higher reflection reward** as the preferred choice.

* 🔗 Constructs (preferred, rejected) pairs from these outputs to serve as **DPO training data**.

> Unlike static or hand-crafted rewards, ReAgent-V’s feedback is **context-aware, multi-dimensional, and fully dynamic**, adapting to each video-question instance.


---

> 🌀 **Unified Insight**:
> ReAgent-V closes the data curation loop. Its multi-agent reward pipeline not only improves current inference but **continuously supplies high-fidelity data** for future optimization—making learning systems **self-refining** in the wild.

- 📁 Module: `ReAgent-V`  
- 📘 Instructions: [Video Understanding README](ReAgent-V/README.md)

  
## 🧑‍💻 Getting Started

Each subfolder contains its own `README.md` with detailed installation, setup, and training instructions. To get started:

1. Clone the repository  
```bash
git clone https://github.com/nta2112/EvoAgent.git
cd EvoAgent
```
2. Follow the environment setup and requirements in each module  
3. Explore the demo scripts and customize as needed

💬 If you have questions or encounter any issues, feel free to open an [issue](https://github.com/nta2112/EvoAgent/issues).

---

## 📚 Citation

If you find ReAgent-V helpful in your research or projects, please consider citing:

```bibtex
@article{zhou2025reagent,
  title={ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding},
  author={Zhou, Yiyang and He, Yangfan and Su, Yaofeng and Han, Siwei and Jang, Joel and Bertasius, Gedas and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2506.01300},
  year={2025}
}
