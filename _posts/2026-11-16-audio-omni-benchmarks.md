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
  .benchmark-citation-toggle, .benchmark-examples-toggle { display: inline; padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; text-align: left; cursor: pointer; }
  .benchmark-citation-row[hidden], .benchmark-examples-row[hidden] { display: none; }
  .benchmark-citation-row td { padding: .75rem; white-space: normal; overflow-wrap: anywhere; font-size: .85rem; line-height: 1.6; }
  .benchmark-example-image { display: block; width: auto; max-width: 100%; height: auto; }
---

<br>
## 1. 音频基准

### 1.1 自动语音识别

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
        <td> 英语 ASR 基准，基于公共领域有声书构建，提供高质量的音频与文本转录对齐数据，并通过 Clean 和 Other 子集评估不同难度下的语音识别性能。</td>
        <td>音频：约 1K 小时<br> 说话人：2,484</td>
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
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="aishell1-citation" title="展开或收起引用">AISHELL-1</button></td>
        <td>2017 年</td>
        <td>中文普通话 ASR 基准，包含不同性别、年龄和地域口音的语音数据，提供人工转录文本及发音词典。</td>
        <td>语音：约 170 小时 <br>说话人：400</td>
        <td>CER</td>
      </tr>
      <tr id="aishell1-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Bu et al. <em>AISHELL-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline.</em> O-COCOSDA, 2017.
          <a href="https://arxiv.org/abs/1709.05522">论文链接</a> ·
          <a href="https://openslr.org/33/">数据集下载</a>。
          <p><b>数据集构成：</b>AISHELL-1 是 AISHELL-ASR0009 普通话语音数据库的开源子集，包含来自中国不同地区的 400 名说话人，男性占 47%、女性占 53%，约 80% 的说话人年龄为 16–25 岁。原始语音通过高保真麦克风、Android 手机和 iPhone 三类设备同步录制，开源版本使用高保真麦克风录音，并统一重采样为 16 kHz、16-bit WAV 格式。语音文本涵盖金融、科技、体育、娱乐和新闻五个领域。数据集按说话人互不重叠划分为训练集（120,098 条、340 人、150 小时）、验证集（14,326 条、40 人、10 小时）和测试集（7,176 条、20 人、5 小时）。</p>
          <p><b>标注形式：</b>每段语音提供人工转录的中文文本，并通过质量检查剔除转录不一致及明显发音错误的样本。文本经过规范化处理，涵盖数字、英文缩写和专有名词等。此外，数据集提供中文发音词典（Lexicon），收录常用汉字、词语及其声韵母发音表示，支持语音识别系统的建模与解码。</p>
          <p><b>评测任务与指标：</b>面向中文自动语音识别（Automatic Speech Recognition, ASR），要求将输入语音转写为对应中文文本。采用字符错误率（Character Error Rate, CER），以替换（Substitution）、删除（Deletion）和插入（Insertion）错误总数除以参考文本字符数计算；CER 越低表示识别效果越好。论文提供 GMM-HMM、TDNN-HMM 和 LF-MMI 三类基线，测试集 CER 分别为 12.23%、8.42% 和 7.62%。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="aishell2-citation" title="展开或收起引用">AISHELL-2</button></td>
        <td>2018 年</td>
        <td>AISHELL-1 的大规模扩展，面向工业级中文普通话语音识别应用。</td>
        <td>语音：约 1K 小时 <br>说话人：1,991</td>
        <td>CER</td>
      </tr>
      <tr id="aishell2-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          <em>AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale.</em>
          <a href="https://arxiv.org/abs/1808.10583">论文链接</a> ·
          <a href="https://www.aishelltech.com/aishell_2">数据集链接</a>。
          <p><b>数据集构成：</b>作为 AISHELL-1 的大规模扩展，数据规模从约 170 小时增至约 1,000 小时，包含约 100 万条语音和 1,991 名说话人。录音由 iOS、Android 和高保真麦克风三类设备同步采集；除录音室外，还包含具有自然混响的客厅环境，以提升模型对不同录音条件的适应能力。</p>
          <p><b>录音设备与评测设置：</b>训练集仅公开 iOS 通道，验证集和测试集提供 iOS、Android 与高保真麦克风三个通道，可分别比较不同设备下的普通话识别性能。</p>
          <p><b>评测指标：</b>沿用 AISHELL-1 的字符错误率（Character Error Rate, CER），根据转写结果中的替换、删除和插入错误总数相对于参考文本字符数计算；CER 越低表示识别效果越好。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>

### 1.2 基础音频感知

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="基础音频感知基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="deaftest-citation" title="展开或收起引用">DeafTest</button></td>
        <td>2024 年</td>
        <td>评估底层声学属性的感知能力，包含声音计数、响度比较、音高比较和时长比较四类任务。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="deaftest-examples" title="展开或收起 DeafTest 样例">[样例]</button></td>
        <td>问题：400<br>任务：4 类</td>
        <td>Accuracy</td>
      </tr>
      <tr id="deaftest-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Gong et al. <em>AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information?</em> arXiv:2412.02611, 2024.
          <a href="https://arxiv.org/abs/2412.02611">论文链接</a> ·
          <a href="https://av-odyssey.github.io/">项目主页</a>。
          <p><b>数据集构成：</b>DeafTest 受到听力学中的 Schwabach Test 和视觉领域 BlindTest 启发，包含四类基础音频感知任务，每类 100 道二选一问题，共 400 道。音频根据预设声学参数构造，并通过随机采样及调整输入顺序降低位置偏差。</p>
          <ul>
            <li><b>Sound Counting（声音计数）：</b>判断音频中声音出现的次数。每段音频包含 3–8 次单调声音，相邻声音之间有明显静音间隔。</li>
            <li><b>Loudness Comparison（响度比较）：</b>判断两段音频中哪段更响；较低响度为 30–60 dB，较高响度为 70–100 dB。</li>
            <li><b>Pitch Comparison（音高比较）：</b>判断两段音频中哪段音高更高；低音高频率为 100–500 Hz，高音高频率为 1,000–2,000 Hz。</li>
            <li><b>Duration Comparison（时长比较）：</b>判断两段音频中哪段持续时间更长；短音频为 1–3 秒，长音频为 4–6 秒。</li>
          </ul>
          <p><b>评测指标：</b>采用 Accuracy 衡量各任务表现。所有问题均为二选一，随机猜测基线为 50%。论文实验中，Gemini 1.5 Pro 在声音计数和时长比较上分别达到 81% 和 84%；所有受测模型在响度比较和音高比较上的准确率均低于 65%。结果显示，高层音频语义理解能力不必然意味着具备良好的底层音频感知能力，而这类感知不足可能成为复杂音视频联合理解与推理的瓶颈。</p>
        </td>
      </tr>
      <tr id="deaftest-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/deaftest-example.png' | relative_url }}" alt="DeafTest 响度比较和音高比较样例，展示音频波形、模型回答与错误标记" loading="lazy">
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 2. 全模态基准

### 2.1 通用全模态理解与推理

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="通用全模态理解与推理基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="omnibench-citation" title="展开或收起引用">OmniBench</button></td>
        <td>2024 年</td>
        <td>面向图像、音频和文本三模态联合理解与推理的基准，强调视觉与听觉信息的互补性，涵盖时空实体理解、因果推理和抽象概念理解。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="omnibench-examples" title="展开或收起 OmniBench 样例">[样例]</button></td>
        <td>问答对：1,142<br>平均音频：9.22 秒<br>最长：30 秒</td>
        <td>Accuracy</td>
      </tr>
      <tr id="omnibench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Li et al. <em>OmniBench: Towards The Future of Universal Omni-Language Models.</em> NeurIPS, 2025（2024 年首次发布）。
          <a href="https://arxiv.org/abs/2409.15272">论文链接</a> ·
          <a href="https://m-a-p.ai/OmniBench">项目主页、代码与数据</a>。
          <p><b>数据集构成：</b>包含 1,142 个由人工构建的四选一问答样本，每个样本由单张静态图像（非视频）、音频片段及文本问题与选项组成。音频包括 Speech（771）、Sound Event（265）和 Music（106）。这种设置减少视觉时序信息干扰，突出音视觉信息的互补理解与推理。</p>
          <p><b>任务类型：</b>包含三大类、八个子任务：Temporal-Spatial Entity（时空实体理解：Object Identification &amp; Description，211；Contextual &amp; Environmental，141）；Causal Inference（因果推理：Story Cause Description，230；Current Action &amp; Activity，251；Future Plot and Purpose Inference，237）；Abstract Concept（抽象概念理解：Identity &amp; Relationship，32；Text &amp; Symbols，25；Count &amp; Quantity，15）。</p>
          <p><b>标注形式与质量控制：</b>每个样本提供四个候选答案、唯一正确答案、音频转录或描述，以及分别对应视觉和音频模态的答案依据（Rationale）。数据经过人工交叉审核与 LLaVA-1.6-34B 自动筛选，并通过 Image + Text、Audio Transcript + Text 和 Text Only 三种模态消融设置，过滤仅凭部分模态即可答对的样本。</p>
          <p><b>评测设置：</b>Omni-Understanding 输入原始图像、音频和文本，评估三模态联合能力；Modality Ablation 移除图像或音频，分析各模态贡献；Textual Approximation 使用人工标注的 Audio Transcript 或 VLM 生成的 Image Caption 替代原始模态，以比较多模态模型、音频模型和纯文本模型的感知与推理表现。</p>
          <p><b>评测指标：</b>使用 Accuracy，即预测选项与正确答案一致的样本比例。所有问题均为四选一，随机猜测准确率为 25%。</p>
          <p><b>配套训练数据：</b>作者另构建 OmniInstruct，用 MSRVTT-QA、AVQA 和 Music-AVQA 2.0 进行静态图像提取与跨模态问答筛选，包含 84,580 条训练样本和 11,525 条验证样本，用于三模态联合指令微调（SFT）。数据规模依据论文 Table 1 和 Table 2，任务分类及质量控制依据 Section 3，评测设置与指标依据 Section 4。</p>
        </td>
      </tr>
      <tr id="omnibench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/omnibench-example.png' | relative_url }}" alt="OmniBench 样例，展示目标识别、情境理解、身份关系、动作活动、故事因果、未来推断、文本符号和数量任务" loading="lazy">
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="av-odyssey-citation" title="展开或收起引用与详细信息">AV-Odyssey Bench</button></td>
        <td>2024  </td>
        <td>相比 OmniBench，扩展了音频属性、应用领域和任务类型，支持文本、图像、视频与音频交错输入。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="av-odyssey-examples" title="展开或收起 AV-Odyssey Bench 样例">[样例]</button></td>
        <td>问题：4,555<br>平均时长：16 秒</td>
        <td>Accuracy</td>
      </tr>
      <tr id="av-odyssey-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Gong et al. <em>AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information?</em> arXiv:2412.02611, 2024.
          <a href="https://arxiv.org/abs/2412.02611">论文链接</a> ·
          <a href="https://av-odyssey.github.io/">项目主页</a>。
          <p><b>数据集构成：</b>包含 4,555 道四选一问题，覆盖 26 类任务和 10 个应用领域，包括音乐、动物、交通、日常生活、场景、动作及语音等。数据来自公开数据集及互联网资源，通过自动化问题构建与人工审核相结合的方式生成；每个问题均包含文本、视觉和音频三种模态。</p>
          <p><b>输入形式：</b>支持多张图像、多段音频及视频与文本交错输入。多图像 + 单音频：2,610 题；单图像 + 多音频：891 题；单图像 + 单音频：434 题；单视频 + 单音频：220 题；单视频 + 多音频：400 题。</p>
          <p><b>评测任务：</b>围绕七个音频相关维度组织 26 类任务：Timbre（音色），如乐器、歌手、动物、材质和交通工具识别；Tone（语气），如语音情感与幽默理解；Melody（旋律），如音乐情感、风格分类及舞蹈/电影与音乐匹配；Space（空间），包括四通道空间音频下的声源角度与距离估计；Time（时间），包括音频定位、音视频同步与动作顺序识别；Hallucination（幻觉），判断特定声音是否真实存在；Intricacy（复杂综合理解），结合视觉与多种音频属性进行动作预测和回溯。</p>
          <p><b>质量控制：</b>使用 InternVL2、Qwen2-VL 等视觉语言模型以及 Qwen2-Audio、SALMONN 等音频语言模型进行单模态测试，过滤可由全部受测视觉模型或全部受测音频模型独立解答的问题，共过滤 2.54%，以降低单模态信息冗余。</p>
          <p><b>评测指标：</b>使用 Accuracy 衡量各任务及整体音视频理解表现；所有题目均为四选一，随机猜测基线为 25%。论文中 GPT-4o Audio Caption 和 Gemini 1.5 Pro 的整体准确率分别为 34.5% 和 30.8%。对 Gemini 1.5 Pro 的错误归因显示，感知错误占 81%，其中音频理解错误占 63%，说明音频感知不足是音视频联合理解的重要瓶颈。</p>
          <p><b>与 OmniBench 的主要区别：</b></p>
          <div class="table-responsive">
            <table class="table table-sm" aria-label="OmniBench 与 AV-Odyssey Bench 对比">
              <thead>
                <tr><th>对比维度</th><th>OmniBench</th><th>AV-Odyssey Bench</th></tr>
              </thead>
              <tbody>
                <tr><td>问题数量</td><td>1,142</td><td>4,555</td></tr>
                <tr><td>视觉输入</td><td>单张图像</td><td>图像或视频</td></tr>
                <tr><td>音频输入</td><td>单音频</td><td>单音频或多音频</td></tr>
                <tr><td>交错输入</td><td>不支持</td><td>支持</td></tr>
                <tr><td>评测覆盖</td><td>音色、时间等</td><td>扩展至语气、旋律、空间、幻觉及复杂综合理解等</td></tr>
              </tbody>
            </table>
          </div>
          <p>以上比较依据 AV-Odyssey 原论文 Table 2，音频属性划分沿用论文分类体系。AV-Odyssey 并非 OmniBench 的直接扩展版本，而是在评测目标相近的基础上，扩展了输入形式、数据规模及音频属性覆盖范围。</p>
        </td>
      </tr>
      <tr id="av-odyssey-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/av-odyssey-example.png' | relative_url }}" alt="AV-Odyssey Bench 样例，展示动物识别、危险场景识别、梗图理解、音乐情感分析、舞蹈音乐匹配、音频距离估计、时间定位、幻觉评估和动作预测等任务" loading="lazy">
        </td>
      </tr>
    </tbody>
  </table>
</div>

### 2.2 情感理解

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
