---
layout: post
title: "大模型测评III：音频与全模态基准"
date: 2026-11-16
permalink: /blog/2026/audio-omni-benchmarks/
description: "梳理主流音频与全模态大模型测评基准，便于查阅参考。"
tags: [大模型, 测评基准]
categories: [blog]
related_posts: false
toc:
  beginning: true
_styles: |
  .benchmark-preview-table table { width: 100%; min-width: 800px; table-layout: fixed; margin-bottom: 0; }
  .benchmark-preview-table thead { background-image: linear-gradient(var(--global-divider-color), var(--global-divider-color)), linear-gradient(var(--global-divider-color), var(--global-divider-color)); background-size: 100% 1px; background-position: top, bottom; background-repeat: no-repeat; }
  .benchmark-preview-table th { white-space: nowrap; text-align: left; vertical-align: middle; padding: .5rem 0; border: 0; }
  .benchmark-preview-table td { vertical-align: top; }
  .benchmark-preview-table tbody tr { background-image: linear-gradient(var(--global-divider-color), var(--global-divider-color)); background-size: 100% 1px; background-position: bottom; background-repeat: no-repeat; }
  .benchmark-data-row td { padding: .4rem 0; vertical-align: top; }
  .benchmark-citation-toggle { display: inline; padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; text-align: left; cursor: pointer; }
  .benchmark-citation-row[hidden] { display: none; }
  .benchmark-citation-row td { padding: .75rem; white-space: normal; overflow-wrap: anywhere; font-size: .85rem; line-height: 1.6; }
---

<br>
## 1. 音频基准

### 自动语音识别（ASR）

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="自动语音识别基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 51%">
      <col style="width: 15%">
      <col style="width: 12%">
    </colgroup>
    <thead>
      <tr>
        <th scope="col">名称</th>
        <th scope="col">发布时间</th>
        <th scope="col">简介</th>
        <th scope="col">规模</th>
        <th scope="col">评测指标</th>
      </tr>
    </thead>
    <tbody>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="librispeech-citation" title="展开或收起引用">LibriSpeech</button></td>
        <td>2015 年</td>
        <td>大规模英语自动语音识别（ASR）基准，基于公共领域有声书构建，提供高质量的音频与文本转录对齐数据，并通过 Clean 和 Other 子集评估不同难度下的语音识别性能。</td>
        <td>音频：约 1,000 小时<br>采样率：16 kHz<br>说话人：2,484</td>
        <td>WER</td>
      </tr>
      <tr id="librispeech-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Panayotov et al. <em>LibriSpeech: An ASR Corpus Based on Public Domain Audio Books.</em> ICASSP, 2015.
          <a href="https://www.clsp.jhu.edu/wp-content/uploads/2015/11/icassp2015_librispeech.pdf">论文链接</a> ·
          <a href="https://www.openslr.org/12/">数据集链接</a>。
          <p><b>数据集构成：</b>基于 LibriVox 项目的英语有声书录音及 Project Gutenberg 的对应书籍文本构建，通过自动语音识别、文本对齐和错误过滤获得音频—文本样本，并排除多人朗读及部分音质较差的录音。训练集包括 train-clean-100（100.6 小时）、train-clean-360（363.6 小时）和 train-other-500（496.7 小时）；评测集包括 dev-clean（5.4 小时）、dev-other（5.3 小时）、test-clean（5.4 小时）和 test-other（5.1 小时）。各子集之间没有说话人重叠。</p>
          <p><b>标注形式：</b>每段音频配有对应的英语文本转录。作者先使用 Smith-Waterman 算法进行音频与文本的初步对齐，再利用包含音素级错误路径的解码图检测插入、删除和替换等转录错误，并过滤不可靠样本。训练集主要依据静音间隔切分音频；验证集和测试集则优先在句子边界切分，以保留完整语言上下文。</p>
          <p><b>评测设置：</b>采用 Clean 和 Other 两类测试集。两类数据依据预训练 ASR 模型的词错误率划分：Clean 包含相对容易识别的语音，Other 包含更具挑战性的口音或录音条件；Other 表示识别难度较高，并不代表存在错误标注。</p>
          <p><b>评测指标：</b>使用 Word Error Rate（WER，词错误率），根据模型转录与参考文本之间的替换错误数（S）、删除错误数（D）和插入错误数（I）计算：<span>WER = (S + D + I) / N</span>，其中 N 为参考文本的单词总数，WER 越低表示识别性能越好。原论文中，使用全部约 960 小时训练数据的 DNN 声学模型结合 4-gram 语言模型，在 test-clean 和 test-other 上分别取得 5.51% 和 13.97% 的 WER。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 2. 全模态基准

### 2.1 情感理解

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="全模态情感理解基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 51%">
      <col style="width: 15%">
      <col style="width: 12%">
    </colgroup>
    <thead>
      <tr>
        <th scope="col">名称</th>
        <th scope="col">发布时间</th>
        <th scope="col">简介</th>
        <th scope="col">规模</th>
        <th scope="col">评测指标</th>
      </tr>
    </thead>
    <tbody>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="dfew-citation" title="展开或收起引用">DFEW</button></td>
        <td>2020 年</td>
        <td>面向真实场景的<b>面部表情识别</b>基准，涵盖复杂光照、面部遮挡和姿态变化等干扰；原始视频音轨可用于扩展音视频情绪识别。</td>
        <td>视频片段：16,372 <br>平均时长：3.42 秒</td>
        <td>UAR / WAR</td>
      </tr>
      <tr id="dfew-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Jiang et al. <em>DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild.</em> ACM MM, 2020.
          <p><b>数据集构成：</b>从全球 1,500 多部高清电影中采集 16,372 个动态面部表情视频片段，覆盖不同场景及复杂自然环境。视频平均时长约 3.42 秒，时长主要集中在 2–5 秒；单标签子集的时长分布为 0–2 秒 4,056 个、2–5 秒 5,734 个、5 秒以上 2,269 个。</p>
          <p><b>标注形式：</b>每个视频由 10 名独立标注者标注，提供 Happy、Sad、Neutral、Angry、Surprise、Disgust 和 Fear 七类情绪的投票分布。对获得超过 6 票一致支持的主导情绪进行筛选，得到 12,059 个单标签样本。</p>
          <p><b>模态扩展：</b>原始论文主要使用 RGB 视频进行动态面部表情识别；原始电影视频也可提取音轨。后续 AVERE（ICLR 2026）将 DFEW 用于音视频情绪识别评测，采用 24 FPS 视频和 16 kHz 单声道音频作为联合输入。</p>
          <p><b>评测指标与划分：</b>UAR（Unweighted Average Recall）衡量各情绪类别的平均识别能力，WAR（Weighted Average Recall）衡量整体识别准确率。实验采用 5-fold Cross-Validation：数据分为五个互不重叠的子集，每次用一个子集测试、其余四个子集训练。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mer2023-citation" title="展开或收起引用">MER 2023</button></td>
        <td>2023 年</td>
        <td>包含三个评测赛道：MER-MULTI 联合预测离散情绪类别与连续情绪效价；MER-NOISE 评估噪声干扰下的模态鲁棒性；MER-SEMI 利用未标注视频进行半监督情绪识别。</td>
        <td>视频片段：78.2K<br>平均时长：3.3 秒</td>
        <td>Weighted F1 / Combined Score</td>
      </tr>
      <tr id="mer2023-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Lian et al. <em>MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning.</em> ACM MM, 2023.
          <a href="https://arxiv.org/abs/2304.08981">论文链接</a>。
          <p><b>数据集构成：</b>基于 CHEAVD 扩展构建，包含 Train&Val（3,373）、MER-MULTI（411）、MER-NOISE（412）和 MER-SEMI（834 个标注样本及 73,148 个未标注样本），共计 78,178 个视频片段。平均时长约 3.3 秒，视频主要来源于中文电影和电视剧中的人物情绪表达片段。</p>
          <p><b>标注形式：</b>提供离散与维度情绪标注。离散情绪包括 Neutral、Anger、Happiness、Sadness、Worry 和 Surprise 六类；维度情绪标注为 Valence 连续分数，表示情绪从消极到积极的程度，并非完整的 Valence-Arousal-Dominance（VAD）三维标注。</p>
          <ul>
            <li><b>MER-MULTI：</b>联合预测离散情绪类别与连续 Valence，考察多标签学习及两类情绪标注之间的关联。</li>
            <li><b>MER-NOISE：</b>在视频模糊、背景噪声等干扰下评估模型的模态鲁棒性。</li>
            <li><b>MER-SEMI：</b>利用 73,148 个未标注视频样本，通过半监督学习提升情绪识别性能。</li>
          </ul>
          <p><b>评测指标：</b>MER-MULTI 和 MER-NOISE 使用 Combined Score：F1<sub>weighted</sub> − 0.25 × MSE<sub>valence</sub>，综合衡量离散情绪分类与连续效价预测；MER-SEMI 仅评估离散情绪识别，采用 Weighted Average F-score。数据规模和视频时长依据论文 Table 1，情绪类别与标注形式依据论文第 2 页。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mer2024-citation" title="展开或收起引用">MER 2024</button></td>
        <td>2024 年</td>
        <td>包含三个评测赛道：MER-SEMI 利用大规模未标注视频进行半监督情绪识别；MER-NOISE 评估音频噪声和图像模糊干扰下的情绪识别鲁棒性；MER-OV 引入开放词汇情绪识别，允许模型自由生成情绪标签。</td>
        <td>视频片段：120.6K<br>MER-OV：332<br>时长：2–16 秒</td>
        <td>WAF / Set-level Accuracy </td>
      </tr>
      <tr id="mer2024-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Lian et al. <em>MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition.</em> MRAC @ ACM MM, 2024.
          <a href="https://arxiv.org/abs/2404.17113">论文链接</a>。
          <p><b>数据集构成：</b>在 MER 2023 基础上扩展，包含 Train&amp;Val（5,030）、MER-SEMI（1,169 个标注样本及 115,595 个视频片段）、MER-NOISE（1,170 个标注样本及 115,595 个视频片段）和 MER-OV（332 个样本）。视频主要来源于中文电影和电视剧，经过视频切分、人脸检测及身份一致性筛选，片段时长为 2–16 秒。</p>
          <p><b>标注形式：</b>MER-SEMI 和 MER-NOISE 使用 Worried、Happy、Neutral、Angry、Surprised 和 Sad 六类离散情绪标签。MER-OV 采用开放词汇标注，不限制情绪类别和标签数量，每个样本平均约有 3 个情绪标签。</p>
          <ul>
            <li><b>MER-SEMI：</b>利用大规模同领域未标注视频开展半监督学习，研究如何在有限标注数据下提升情绪识别性能。</li>
            <li><b>MER-NOISE：</b>在音频中加入加性噪声（SNR 为 5–10 dB），并对视频图像进行模糊处理，评估复杂环境下的噪声鲁棒性。</li>
            <li><b>MER-OV：</b>允许模型生成任意数量、任意类别的情绪标签，以更全面地描述人物情绪，避免局限于预定义类别。</li>
          </ul>
          <p><b>评测指标：</b>MER-SEMI 和 MER-NOISE 使用 Accuracy 和 Weighted Average F1-score（WAF），其中 WAF 为最终排名指标。MER-OV 使用 Set-level Accuracy 和 Set-level Recall，并以二者平均值作为最终指标；计算前会通过语义归并处理不同表达但含义相同的情绪标签。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="emer-citation" title="展开或收起引用">EMER</button></td>
        <td>2023 年</td>
        <td>基于 MER 2023 的中文影视片段构建可解释多模态情绪识别任务，要求模型结合视觉、音频和文本线索生成情绪解释，并识别开放词汇情绪标签。</td>
        <td>视频片段：332<br>平均时长：3.78 秒</td>
        <td>Accuracy / Recall / Avg</td>
      </tr>
      <tr id="emer-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Lian et al. <em>Explainable Multimodal Emotion Recognition.</em> arXiv:2306.15401, 2023（2024 年修订）。
          <a href="https://arxiv.org/abs/2306.15401">论文链接</a>。
          <p><b>数据集构成：</b>从 MER 2023 中选取 332 个中文影视视频片段，包含视频、音频和字幕三种模态。先由 GPT-4V 生成视觉线索描述并经两轮人工校验，再由 GPT-3.5 结合多模态线索消除字幕歧义，构建可解释情绪描述（EMER Description）。平均视频时长为 3.78 秒。</p>
          <p><b>标注形式：</b>情绪解释涵盖面部表情、身体动作、声学特征及文本语义，并从中提取开放词汇情绪标签。不同于传统六类离散情绪标注，EMER 不限制情绪类别，共有 301 种候选标签，每个样本平均约 3 个标签。</p>
          <p><b>评测指标：</b>将模型生成的情绪描述转换为开放词汇情绪标签集合，经同义词归并后计算集合级 Accuracy 和 Recall，并以两者的算术平均值 Avg 作为主要排名指标。另提供 BLEU-1、BLEU-4、METEOR 和 ROUGE-L 衡量生成描述与参考描述的文本相似性。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ov-merd-citation" title="展开或收起引用">OV-MERD</button></td>
        <td>2025 年</td>
        <td>开放词汇多模态情绪识别数据集，与 EMER 和 MER 2024 的 MER-OV 使用相同的 332 个视频样本。以集合级 F1 作为主要指标，并提供基于 Emotion Wheel 的语义归并方法，降低对 GPT API 的依赖。</td>
        <td>视频片段：332<br>情绪类别：236<br>时长主要为 1–4 秒</td>
        <td>Set-level Precision / Recall / F1</td>
      </tr>
      <tr id="ov-merd-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          <em>OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition.</em> ICML, 2025.
          <a href="https://arxiv.org/abs/2410.01495">论文链接</a> · <a href="https://github.com/zeroQiaoba/AffectGPT">代码与数据</a>。
          <p><b>数据集构成：</b>与 EMER（2023）和 MER 2024 的 MER-OV 使用相同的 332 个视频样本，但对标注和评测方式进行了改进。最终覆盖 236 种情绪类别，每个样本包含 1–9 个情绪标签，多数为 2–4 个；视频时长主要集中在 1–4 秒。</p>
          <ul>
            <li><b>标注：</b>EMER 侧重情绪解释，MER-OV 沿用其开放词汇标签；OV-MERD 通过人类与大模型协作重新整理标注。</li>
            <li><b>评测：</b>EMER 和 MER-OV 使用集合级 Precision（原文称 Accuracy）与 Recall 的算术平均作为排名指标；OV-MERD 改用集合级 F1（调和平均），更强调准确性与覆盖率的平衡。</li>
            <li><b>语义匹配：</b>EMER 和 MER-OV 主要使用 GPT 归并相近情绪；OV-MERD 进一步提供基于 Emotion Wheel 的归并方法，降低对 GPT API 的依赖。</li>
          </ul>
          <p><b>评测指标：</b>Set-level Precision、Recall 和 F1，其中 F1 为主要指标。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
