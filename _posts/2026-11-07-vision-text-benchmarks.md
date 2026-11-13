---
layout: post
title: "大模型测评基准II：视觉文本基准"
date: 2026-11-07
permalink: /blog/2026/vision-text-benchmarks/
description: "梳理主流视觉文本基准，便于个人查阅参考。"
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
  .benchmark-data-row td { padding-top: .4rem; padding-bottom: .4rem; }
  .benchmark-data-row td:not(:nth-child(3)) { white-space: nowrap; }
  .benchmark-data-row td.benchmark-count-cell { white-space: normal; }
  .benchmark-data-row td.benchmark-metric-cell { white-space: normal; }
  .benchmark-citation-toggle, .benchmark-datasets-toggle, .benchmark-performance-toggle, .benchmark-examples-toggle { display: inline; padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; font-weight: normal; text-decoration: none; white-space: normal; overflow-wrap: anywhere; text-align: left; cursor: pointer; }
  .benchmark-citation-toggle { max-width: 100%; white-space: normal; text-align: left; }
  .benchmark-citation-row[hidden], .benchmark-datasets-row[hidden], .benchmark-performance-row[hidden], .benchmark-examples-row[hidden] { display: none; }
  .benchmark-citation-row td, .benchmark-datasets-row td { padding: .75rem; white-space: normal; overflow-wrap: anywhere; font-size: .85rem; line-height: 1.6; }
  .benchmark-dataset-list { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .35rem 1rem; margin: 0; padding: 0; list-style: none; }
  .benchmark-performance-row td { padding: .75rem; white-space: normal; }
  .benchmark-performance-panel, .benchmark-task-list-panel { position: sticky; left: 0; width: min(100%, calc(100vw - 60px)); }
  @media (max-width: 575px) { .benchmark-task-list-panel { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .benchmark-performance-chart { width: 100%; height: 430px; }
  .benchmark-performance-status { margin: .5rem 0; font-size: .85rem; }
  .benchmark-examples-row td { padding: .75rem; white-space: normal; overflow-wrap: anywhere; }
  .benchmark-examples-panel { position: sticky; left: 0; width: min(100%, calc(100vw - 60px)); font-size: .95rem; line-height: 1.65; }
  .benchmark-example-image { display: block; width: auto; max-width: 100%; height: auto; }
  .benchmark-example { padding: .75rem 0; border-bottom: 1px solid var(--global-divider-color); }
  .benchmark-example:first-child { padding-top: 0; }
  .benchmark-example:last-child { padding-bottom: 0; border-bottom: 0; }
  .benchmark-example p { margin-bottom: .4rem; }
  .benchmark-example p:last-child { margin-bottom: 0; }
  .benchmark-example .highlight { margin: 0; max-width: 100%; }
  .benchmark-example pre { max-width: 100%; overflow-x: auto; white-space: pre; }
  .benchmark-example-correct { background-color: rgba(97, 169, 189, .25); }
  .benchmark-example-result-correct { color: #198754; font-weight: bold; }
  .benchmark-example-result-incorrect { color: #bd4147; font-weight: bold; }
  .benchmark-preview-table .benchmark-subtask-table { width: 100%; min-width: 0; margin: 0; table-layout: fixed; }
  .benchmark-subtask-table th, .benchmark-subtask-table td { padding: .4rem .55rem; border: 0; border-bottom: 1px solid var(--global-divider-color); text-align: left; vertical-align: top; white-space: normal; }
  .benchmark-subtask-table th:first-child { width: 22%; }
  .benchmark-subtask-table th:nth-child(2) { width: 38%; }
  .benchmark-subtask-table th:nth-child(3) { width: 40%; }
  .benchmark-subtask-table.seed-bench-dimensions-table th:first-child { width: 32%; }
  .benchmark-subtask-table.seed-bench-dimensions-table th:nth-child(2) { width: 20%; }
  .benchmark-subtask-table.seed-bench-dimensions-table th:nth-child(3) { width: 48%; }
  .benchmark-subtask-table.okvqa-categories-table th:first-child { width: 8%; }
  .benchmark-subtask-table.okvqa-categories-table th:nth-child(2) { width: 55%; }
  .benchmark-subtask-table.okvqa-categories-table th:nth-child(3) { width: 37%; }
  .benchmark-subtask-table.mmmu-disciplines-table th:first-child { width: 28%; }
  .benchmark-subtask-table.mmmu-disciplines-table th:nth-child(2) { width: 10%; }
  .benchmark-subtask-table.mmmu-disciplines-table th:nth-child(3) { width: 62%; }
  .benchmark-subtask-table tbody:last-child tr:last-child th, .benchmark-subtask-table tbody:last-child tr:last-child td { border-bottom: 0; }
---

<br>
## 1. 综合视觉理解


<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="综合视觉理解基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mme-citation" title="展开或收起引用">MME</button></td>
        <td>2023 年</td>
        <td>包含<b>感知和认知</b>两大维度、共 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mme-subtasks" title="展开或收起14项子任务">14 项子任务</button>。所有题目均采用 <b>Yes/No 二分类</b>形式，每张图像对应一正一负两道问题，最终以汇总分数 <b>Score</b> 衡量模型表现。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mme-examples" title="展开或收起MME文字样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：1.2K<br>问题：2.4K</td>
        <td class="benchmark-metric-cell">Score</td>
      </tr>
      <tr id="mme-subtasks" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table" aria-label="MME 的 14 项子任务">
            <caption>MME 的 14 项子任务</caption>
            <thead>
              <tr>
                <th scope="col">类别</th>
                <th scope="col">子任务</th>
                <th scope="col">中文名称</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="rowgroup" rowspan="10">Perception</th>
                <td>Existence</td>
                <td>物体存在性识别</td>
              </tr>
              <tr><td>Count</td><td>物体计数</td></tr>
              <tr><td>Position</td><td>物体位置识别</td></tr>
              <tr><td>Color</td><td>颜色识别</td></tr>
              <tr><td>OCR</td><td>光学字符识别</td></tr>
              <tr><td>Poster</td><td>电影海报识别</td></tr>
              <tr><td>Celebrity</td><td>名人识别</td></tr>
              <tr><td>Scene</td><td>场景识别</td></tr>
              <tr><td>Landmark</td><td>地标识别</td></tr>
              <tr><td>Artwork</td><td>艺术作品识别</td></tr>
            </tbody>
            <tbody>
              <tr>
                <th scope="rowgroup" rowspan="4">Cognition</th>
                <td>Commonsense Reasoning</td>
                <td>常识推理</td>
              </tr>
              <tr><td>Numerical Calculation</td><td>数值计算</td></tr>
              <tr><td>Text Translation</td><td>文本翻译</td></tr>
              <tr><td>Code Reasoning</td><td>代码推理</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mme-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Fu et al. <em>MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models.</em> arXiv:2306.13394, 2023。<a href="https://arxiv.org/abs/2306.13394">论文链接</a>。
          <p>最终得分为 Accuracy 与 Accuracy+ 的总和（Score = Accuracy + Accuracy+，Sum）；Accuracy 是单题准确率，Accuracy+ 是同一图像对应两题均答对的比例。</p>
          <table class="benchmark-subtask-table" aria-label="MME 各评测维度的最高分">
            <thead>
              <tr><th scope="col">评测维度</th><th scope="col">子任务数量</th><th scope="col">最高分</th></tr>
            </thead>
            <tbody>
              <tr><td>Perception（感知）</td><td>10</td><td>2,000</td></tr>
              <tr><td>Cognition（认知）</td><td>4</td><td>800</td></tr>
              <tr><td>Overall（合计）</td><td>14</td><td>2,800</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mme-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mme-example.png' | relative_url }}" alt="MME 样例截图，展示感知、OCR 与认知任务中的 Yes/No 问题" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmbench-citation" title="展开或收起引用">MMBench</button></td>
        <td>2023 年</td>
        <td>包含<b>感知和推理</b>两大维度，<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mmbench-dimensions" title="展开或收起20个评测维度">20 个评测维度</button>，涵盖图像场景理解、物体定位、属性识别、空间关系与逻辑推理。采用人工构建的多项选择题，提供中英文两个版本。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mmbench-examples" title="展开或收起MMBench样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：3.2K<br>问题：3.2K</td>
        <td class="benchmark-metric-cell">CircularEval Accuracy</td>
      </tr>
      <tr id="mmbench-dimensions" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table" aria-label="MMBench 的 20 个评测维度">
            <caption>MMBench 的 20 个评测维度</caption>
            <thead>
              <tr><th scope="col">一级维度</th><th scope="col">二级维度</th><th scope="col">三级评测能力</th></tr>
            </thead>
            <tbody>
              <tr>
                <th scope="rowgroup" rowspan="3">Perception（感知）</th>
                <td>Coarse Perception</td>
                <td>Image Style、Image Scene、Image Emotion、Image Quality、Image Topic</td>
              </tr>
              <tr>
                <td>Fine-grained Perception (Single-instance)</td>
                <td>Object Localization、Attribute Recognition、Celebrity Recognition、OCR</td>
              </tr>
              <tr>
                <td>Fine-grained Perception (Cross-instance)</td>
                <td>Spatial Relationship、Attribute Comparison、Action Recognition</td>
              </tr>
              <tr>
                <th scope="rowgroup" rowspan="3">Reasoning（推理）</th>
                <td>Attribute Reasoning</td>
                <td>Physical Property Reasoning、Function Reasoning、Identity Reasoning</td>
              </tr>
              <tr>
                <td>Relation Reasoning</td>
                <td>Social Relation、Nature Relation、Physical Relation</td>
              </tr>
              <tr>
                <td>Logic Reasoning</td>
                <td>Structuralized Image-Text Understanding、Future Prediction</td>
              </tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mmbench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Liu et al. <em>MMBench: Is Your Multi-modal Model an All-around Player?</em> arXiv:2307.06281, 2023。
          <a href="https://arxiv.org/abs/2307.06281">论文链接</a>。
          <p><b>评测方法：</b>CircularEval 将一道含 N 个选项的问题循环移位为 N 种顺序；模型须在所有轮次都答对，该题才计为正确。自由文本回答无法直接提取选项时，使用 GPT-4 匹配对应选项，以减少严格字符串匹配造成的误判。</p>
        </td>
      </tr>
      <tr id="mmbench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mmbench-example.png' | relative_url }}" alt="MMBench 样例截图，展示英文与中文版本的磁铁间磁力选择题" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="realworldqa-citation" title="展开或收起引用">RealWorldQA</button></td>
        <td>2024 年</td>
        <td>评估对真实物理环境的<b>空间理解能力</b>，涵盖物体相对位置、距离判断、方向识别、物体大小比较及道路场景理解等。数据包含<b>车辆拍摄</b>的匿名化图像及其他真实场景图像。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="realworldqa-examples" title="展开或收起RealWorldQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：0.8K<br>问题：0.8K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="realworldqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          来源：xAI, <em>Grok-1.5 Vision Preview</em>, 2024（官方博客发布，非独立学术论文）。
        </td>
      </tr>
      <tr id="realworldqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/realworldqa-example.png' | relative_url }}" alt="RealWorldQA 样例截图，夜间道路场景下询问当前车道可行驶方向" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="seed-bench-citation" title="展开或收起引用">SEED-Bench</button></td>
        <td>2023 年</td>
        <td> 包含<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="seed-bench-dimensions" title="展开或收起12个评测维度">12 个评测维度</button>。采用人工验证的<b>四选一选择题</b>，评估模型的视觉感知、空间推理与时序理解能力。相比MME，SEED-Bench 样本数量更多，覆盖范围更广。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="seed-bench-examples" title="展开或收起SEED-Bench样例">[样例]</button></td>
        <td class="benchmark-count-cell">问题：19K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="seed-bench-dimensions" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table seed-bench-dimensions-table" aria-label="SEED-Bench 的 12 个评测维度">
            <caption>SEED-Bench 的 12 个评测维度</caption>
            <thead>
              <tr><th scope="col">评测维度</th><th scope="col">中文名称</th><th scope="col">评测内容</th></tr>
            </thead>
            <tbody>
              <tr><td colspan="3"><b>一、空间理解（Spatial Understanding）：9 个</b></td></tr>
              <tr><td>1. Scene Understanding</td><td>场景理解</td><td>理解图像整体场景和语义</td></tr>
              <tr><td>2. Instance Identity</td><td>物体识别</td><td>识别物体类别或判断其是否存在</td></tr>
              <tr><td>3. Instance Attribute</td><td>物体属性</td><td>识别物体的颜色、形状、材质等</td></tr>
              <tr><td>4. Instance Location</td><td>物体定位</td><td>判断指定物体在图像中的位置</td></tr>
              <tr><td>5. Instance Counting</td><td>物体计数</td><td>统计指定类别物体的数量</td></tr>
              <tr><td>6. Spatial Relation</td><td>空间关系</td><td>判断物体之间的相对位置</td></tr>
              <tr><td>7. Instance Interaction</td><td>物体交互</td><td>理解人物或物体之间的交互关系</td></tr>
              <tr><td>8. Visual Reasoning</td><td>视觉推理</td><td>结合图像信息与常识进行推理</td></tr>
              <tr><td>9. Text Recognition</td><td>文字识别</td><td>识别并理解图像中的文字</td></tr>
            </tbody>
            <tbody>
              <tr><td colspan="3"><b>二、时序理解（Temporal Understanding）：3 个</b></td></tr>
              <tr><td>10. Action Recognition</td><td>动作识别</td><td>识别视频中的具体动作</td></tr>
              <tr><td>11. Action Prediction</td><td>动作预测</td><td>根据已有视频预测后续动作</td></tr>
              <tr><td>12. Procedure Understanding</td><td>过程理解</td><td>识别视频中的关键动作及其先后顺序</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="seed-bench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Li et al. <em>SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension.</em> arXiv:2307.16125, 2023.
          <a href="https://arxiv.org/abs/2307.16125">论文链接</a> ·
          <a href="https://github.com/AILab-CVC/SEED-Bench">官方仓库</a>。
          <p>注：规模采用原论文 Table 1 的 19,242 道选择题，包含 9 个图像空间理解维度与 3 个视频时序理解维度。原论文采用候选答案似然排序确定预测选项，并以 Accuracy 评估。</p>
        </td>
      </tr>
      <tr id="seed-bench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/seed-bench-example.png' | relative_url }}" alt="SEED-Bench 样例截图，展示图像空间理解与视频时序理解问题" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="scienceqa-citation" title="展开或收起引用">ScienceQA</button></td>
        <td>2022 年</td>
        <td>面向小学至高中科学教育的多模态问答基准，采用多项选择题，覆盖自然科学、社会科学和语言科学，还提供背景知识和答案解释。</td>
        <td class="benchmark-count-cell">图像：7.8K<br>问题：21.2K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="scienceqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Lu et al. <em>Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering.</em> NeurIPS, 2022. arXiv:2209.09513. <a href="https://arxiv.org/abs/2209.09513">论文链接</a>。
          <p><b>数据集构成：</b>共 21,208 道题目，涵盖 3 个学科、26 个主题（Topics）、127 个类别（Categories）、379 项具体技能（Skills）及 1–12 年级。Train / Val / Test 分别为 12,726 / 4,241 / 4,241 道。</p>
          <p><b>上下文：</b>10,332 道（48.7%）包含图像，10,220 道（48.2%）包含文本；其中 6,532 道（30.8%）同时包含图像和文本，7,188 道（33.9%）不含额外上下文。图像包括自然图像与科学示意图。</p>
          <p><b>标注与评测：</b>17,798 道（83.9%）提供 Lecture，19,202 道（90.5%）提供 Explanation。主指标为 Accuracy；解释生成质量还可用 BLEU、ROUGE-L、Sentence-BERT Similarity 及人工评估（相关性、正确性、完整性）衡量。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmstar-citation" title="展开或收起引用">MMStar</button></td>
        <td>2024 年</td>
        <td>从现有多模态基准收集候选样本，减少无需图像即可作答的问题及训练数据泄漏。包含<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mmstar-dimensions" title="展开或收起6大核心能力与18个细分维度">6 大核心能力与 18 个细分维度</button>。</td>
        <td class="benchmark-count-cell">样本：1.5K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="mmstar-dimensions" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table" aria-label="MMStar 的六大核心能力与18个细分评测维度">
            <caption>MMStar 的六大核心能力与 18 个细分评测维度</caption>
            <thead>
              <tr><th scope="col">核心能力</th><th scope="col">英文名称</th><th scope="col">细分评测维度</th></tr>
            </thead>
            <tbody>
              <tr><td>粗粒度感知</td><td>Coarse Perception (CP)</td><td>Image Style &amp; Quality（图像风格与质量）、Image Scene &amp; Topic（场景与主题）、Image Emotion（图像情感）</td></tr>
              <tr><td>细粒度感知</td><td>Fine-grained Perception (FP)</td><td>Attribute &amp; Celebrity Recognition（属性与名人识别）、Object Location（物体定位）、Object Counting（物体计数）</td></tr>
              <tr><td>实例推理</td><td>Instance Reasoning (IR)</td><td>Single-instance Attribute Reasoning（单实例属性推理）、Cross-instance Attribute Comparison（跨实例属性比较）、Cross-instance Relation Reasoning（跨实例关系推理）</td></tr>
              <tr><td>逻辑推理</td><td>Logical Reasoning (LR)</td><td>Diagram Reasoning（图表推理）、Code &amp; Sequence Reasoning（代码与序列推理）、Common Reasoning（常识推理）</td></tr>
              <tr><td>科学与技术</td><td>Science &amp; Technology (ST)</td><td>Natural Science（自然科学）、Engineering（工程学）、Geography &amp; Earth Science（地理与地球科学）</td></tr>
              <tr><td>数学能力</td><td>Mathematics (MA)</td><td>Numeric Commonsense &amp; Calculation（数值常识与计算）、Geometry（几何）、Statistical Analysis（统计分析）</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mmstar-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Chen et al. <em>Are We on the Right Way for Evaluating Large Vision-Language Models?</em> arXiv:2403.20330, 2024. <a href="https://arxiv.org/abs/2403.20330">论文链接</a>。
          <p><b>数据构建：</b>从 MMBench、SEED-Bench、MMMU、ScienceQA、AI2D 和 MathVista 收集 22,401 条候选样本，经 8 个 LLM 检查器过滤后保留 11,607 条，再结合 16 个 LVLM 的答题表现划分难度，并由 3 位专家审核，最终选出 1,500 条样本。六大核心能力各含 250 条，并尽可能均衡覆盖 18 个细分维度。构建原则包括视觉依赖性、最小化数据泄漏和高级多模态能力。</p>
          <p><b>评测指标：</b>Accuracy 用于衡量整体及各维度表现。辅助指标包括 Multi-modal Gain（MG = Sᵥ − S₍wv₎，衡量有图像与无图像输入的准确率差异）和 Multi-modal Leakage（ML = max(0, S₍wv₎ − Sₜ)，估计多模态训练可能引入的数据泄漏；Sₜ 为对应原始语言模型准确率）。</p>
          <p><b>评测设置：</b>纯 LLM 使用 2-shot，LVLM 使用 0-shot。论文实验中，纯 LLM 平均准确率为 24.6%，GPT-4V（High）最高为 57.1%；MMStar 平均 ML 为 1.9。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 2. 多学科知识

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="多学科知识基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmmu-citation" title="展开或收起引用">MMMU</button></td>
        <td>2024 年</td>
        <td>面向<b>专家级多模态理解与推理</b>的综合基准，收集大学考试、测验和教材中的多模态问题，覆盖 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mmmu-disciplines" title="展开或收起6大学科及30个专业">6 大学科</button>、30 个专业和 183 个子领域。包含图表、示意图、地图、化学结构、乐谱、医学影像等 <b>30 种异构图像类型</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mmmu-examples" title="展开或收起MMMU样例">[样例]</button></td>
        <td class="benchmark-count-cell">问题：11.5K</td>
        <td class="benchmark-metric-cell">Micro-averaged Accuracy</td>
      </tr>
      <tr id="mmmu-disciplines" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table mmmu-disciplines-table" aria-label="MMMU 六大学科、占比及30个专业">
            <thead>
              <tr><th scope="col">学科</th><th scope="col">占比</th><th scope="col">30 个专业</th></tr>
            </thead>
            <tbody>
              <tr><td>Art &amp; Design（艺术与设计）</td><td>11%</td><td>Art（艺术）、Design（设计）、Music（音乐）、Art Theory（艺术理论）</td></tr>
              <tr><td>Business（商业）</td><td>14%</td><td>Accounting（会计）、Economics（经济学）、Finance（金融）、Management（管理）、Marketing（市场营销）</td></tr>
              <tr><td>Science（科学）</td><td>23%</td><td>Biology（生物学）、Chemistry（化学）、Geography（地理学）、Math（数学）、Physics（物理学）</td></tr>
              <tr><td>Health &amp; Medicine（健康与医学）</td><td>17%</td><td>Basic Medical Science（基础医学）、Clinical Medicine（临床医学）、Diagnostics（医学诊断）、Pharmacy（药学）、Public Health（公共卫生）</td></tr>
              <tr><td>Humanities &amp; Social Science（人文与社会科学）</td><td>9%</td><td>History（历史）、Literature（文学）、Psychology（心理学）、Sociology（社会学）</td></tr>
              <tr><td>Tech &amp; Engineering（技术与工程）</td><td>26%</td><td>Agriculture（农业）、Architecture Engineering（建筑工程）、Computer Science（计算机科学）、Electronics（电子学）、Energy Power（能源动力）、Materials（材料）、Mechanical Engineering（机械工程）</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mmmu-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Yue et al. <em>MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.</em> arXiv:2311.16502, 2024.
          <a href="https://arxiv.org/abs/2311.16502">论文链接</a> ·
          <a href="https://mmmu-benchmark.github.io/">项目主页</a>。
          <p><b>数据划分：</b>Dev / Val / Test = 150 / 900 / 10,500 道题。</p>
          <p><b>题目形式：</b>Multiple-choice Questions 10,861 道（94.03%）；Open Questions 689 道（5.97%）。</p>
          <p><b>难度分布：</b>Easy 28%、Medium 45%、Hard 27%。</p>
          <p><b>评测指标：</b>Micro-averaged Accuracy，按样本数汇总各题的正确率。</p>
        </td>
      </tr>
      <tr id="mmmu-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mmmu-example.png' | relative_url }}" alt="MMMU 样例截图，展示艺术与设计、商业、科学、健康与医学、人文与社会科学、技术与工程六大学科题目" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmmu-pro-citation" title="展开或收起引用">MMMU-Pro</button></td>
        <td>2025 年</td>
        <td>在 MMMU 基础上<b>过滤可仅凭文本解答的问题</b>，将<b>选项从 4 个扩充至 10 个</b>，并引入 <b>Vision-only</b> 输入：把问题和选项嵌入截图或照片，要求模型依据视觉内容与专业知识推理。</td>
        <td class="benchmark-count-cell">图像：1.73K<br>问题：1.73K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="mmmu-pro-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Yue et al. <em>MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark.</em> ACL, 2025.
          <a href="https://aclanthology.org/2025.acl-long.736/">论文链接</a> ·
          <a href="https://mmmu-benchmark.github.io/#leaderboard">项目主页</a>。
          <p><b>评测设置：</b>Standard 将选择题扩充至 10 个选项；Vision-only 将问题与选项嵌入截图或照片，以纯视觉形式输入。</p>
          <p><b>综合成绩：</b>Standard 与 Vision-only 两种设置的 Accuracy 平均值。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 3. 数学能力

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="数据能力基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mathvista-citation" title="展开或收起引用">MathVista</button></td>
        <td>2024 年</td>
        <td>强调<b>视觉上下文的多样性</b>，涵盖几何图形、函数图像、统计图表、自然图像和抽象场景等。包含<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mathvista-tasks" title="展开或收起五类主要任务">五类主要任务</button>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mathvista-examples" title="展开或收起MathVista样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：5.5K<br>问题：6.1K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="mathvista-tasks" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table" aria-label="MathVista 的五类主要任务">
            <caption>MathVista 的五类主要任务</caption>
            <thead>
              <tr><th scope="col">任务类型</th><th scope="col">简称</th><th scope="col">任务说明</th></tr>
            </thead>
            <tbody>
              <tr><td>Figure Question Answering</td><td>FQA</td><td>图表与统计图形问答</td></tr>
              <tr><td>Geometry Problem Solving</td><td>GPS</td><td>几何问题求解</td></tr>
              <tr><td>Math Word Problem</td><td>MWP</td><td>数学应用题</td></tr>
              <tr><td>Textbook Question Answering</td><td>TQA</td><td>教材与科学知识问答</td></tr>
              <tr><td>Visual Question Answering</td><td>VQA</td><td>视觉场景问答</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mathvista-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Lu et al. <em>MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts.</em> ICLR, 2024.
          <a href="https://arxiv.org/abs/2310.02255">论文链接</a> ·
          <a href="https://mathvista.github.io/">项目主页</a>。
          <p><b>数据来源：</b>整合 31 个数据集，包括 9 个数学专项问答数据集、19 个通用视觉问答数据集，以及新构建的 IQTest、FunctionQA 和 PaperQA；新增数据集共 736 道题。</p>
          <p><b>数据划分：</b>Testmini（1,000 道）用于模型开发与验证；Test（5,141 道）用于标准评测，答案不公开，通过在线评测平台评估。</p>
        </td>
      </tr>
      <tr id="mathvista-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mathvista-example.png' | relative_url }}" alt="MathVista 样例截图，展示 IQTest 数字谜题、FunctionQA 函数图像问题和 PaperQA 图表推理问题" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="math-vision-citation" title="展开或收起引用">MATH-Vision</button></td>
        <td>2024 年</td>
        <td>基于<b>真实数学竞赛题</b>的视觉数学推理基准，覆盖<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="math-vision-subjects" title="展开或收起16个数学领域">16 个数学领域</button>和 5 个难度等级。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="math-vision-examples" title="展开或收起MATH-Vision样例">[样例]</button></td>
        <td class="benchmark-count-cell">问题：3.0K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="math-vision-subjects" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table" aria-label="MATH-Vision 的 16 个数学领域">
            <caption>MATH-Vision 的 16 个数学领域</caption>
            <thead>
              <tr><th scope="col">序号</th><th scope="col">英文领域</th><th scope="col">中文名称</th></tr>
            </thead>
            <tbody>
              <tr><th scope="row">1</th><td>Transformation Geometry</td><td>变换几何</td></tr>
              <tr><th scope="row">2</th><td>Algebra</td><td>代数</td></tr>
              <tr><th scope="row">3</th><td>Analytic Geometry</td><td>解析几何</td></tr>
              <tr><th scope="row">4</th><td>Topology</td><td>拓扑学</td></tr>
              <tr><th scope="row">5</th><td>Arithmetic</td><td>算术</td></tr>
              <tr><th scope="row">6</th><td>Combinatorial Geometry</td><td>组合几何</td></tr>
              <tr><th scope="row">7</th><td>Statistics</td><td>统计学</td></tr>
              <tr><th scope="row">8</th><td>Solid Geometry</td><td>立体几何</td></tr>
              <tr><th scope="row">9</th><td>Metric Geometry - Length</td><td>度量几何（长度）</td></tr>
              <tr><th scope="row">10</th><td>Metric Geometry - Area</td><td>度量几何（面积）</td></tr>
              <tr><th scope="row">11</th><td>Metric Geometry - Angle</td><td>度量几何（角度）</td></tr>
              <tr><th scope="row">12</th><td>Combinatorics</td><td>组合数学</td></tr>
              <tr><th scope="row">13</th><td>Counting</td><td>计数</td></tr>
              <tr><th scope="row">14</th><td>Descriptive Geometry</td><td>画法几何</td></tr>
              <tr><th scope="row">15</th><td>Graph Theory</td><td>图论</td></tr>
              <tr><th scope="row">16</th><td>Logic</td><td>逻辑</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="math-vision-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Wang et al. <em>Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset.</em> arXiv:2402.14804, 2024.
          <a href="https://arxiv.org/abs/2402.14804">论文链接</a> ·
          <a href="https://mathvision-cuhk.github.io/">项目主页</a>。
          <p>数据集包含 3,040 道来自真实数学竞赛、带有视觉上下文的题目，覆盖 16 个数学学科领域与 5 个难度等级。</p>
          <p><b>与 MathVista 的区别：</b>MathVista 汇集多种来源的数据，覆盖图表、自然图像等多类视觉数学任务；MATH-Vision 则集中于真实竞赛数学题，更强调从题目图形中提取信息并完成数学推理。</p>
        </td>
      </tr>
      <tr id="math-vision-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/math-vision-example.png' | relative_url }}" alt="MATH-Vision 样例截图，给出等腰三角形图形与几何条件，要求求角 CDE" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mathverse-citation" title="展开或收起引用">MathVerse</button></td>
        <td>2024 年</td>
        <td>与 MATH-Vision 的竞赛题不同，MathVerse 聚焦高中数学的<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mathverse-subjects" title="展开或收起三大数学领域及12个子领域">三大领域</button>，排除微积分、图论等大学知识。其特点是为每道题构建<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mathverse-versions" title="展开或收起六种不同的配置">六种问题</button>，通过逐步减少文本冗余、增加图像信息，考察模型是否真正理解数学图形，而非依赖文本完成推理。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mathverse-examples" title="展开或收起MathVerse样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：2.4K<br>问题：15.7K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="mathverse-subjects" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table" aria-label="MathVerse 的三大数学领域及 12 个子领域">
            <caption>MathVerse 的三大数学领域及 12 个子领域</caption>
            <colgroup><col style="width: 28%"><col style="width: 72%"></colgroup>
            <thead>
              <tr><th scope="col">数学领域</th><th scope="col">子领域</th></tr>
            </thead>
            <tbody>
              <tr><td>Plane Geometry（平面几何）</td><td>Length、Area、Angle、Analytic Geometry、Applied Geometry</td></tr>
              <tr><td>Solid Geometry（立体几何）</td><td>Length、Area、Volume</td></tr>
              <tr><td>Functions（函数）</td><td>Coordinate、Property、Expression、Applied Function</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mathverse-versions" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <ul class="benchmark-dataset-list" aria-label="MathVerse 的六种文本与视觉信息配置">
            <li>Text-dominant</li>
            <li>Text-lite</li>
            <li>Text-only</li>
            <li>Vision-intensive</li>
            <li>Vision-dominant</li>
            <li>Vision-only</li>
          </ul>
        </td>
      </tr>
      <tr id="mathverse-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Zhang et al. <em>MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?</em> arXiv:2403.14624, 2024.
          <a href="https://arxiv.org/abs/2403.14624">论文链接</a>。
          <p><b>知识范围：</b>聚焦高中阶段的平面几何、立体几何和函数问题，不涉及微积分、图论等高等院校高级学科知识。</p>
        </td>
      </tr>
      <tr id="mathverse-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mathverse-example.png' | relative_url }}" alt="MathVerse 样例图，展示同一道圆几何题在 Text Dominant、Text Lite、Text Only、Vision Intensive、Vision Dominant 和 Vision Only 六种配置中的呈现方式" loading="lazy">
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 4. 文本与图表理解

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="文本、图表与示意图理解基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ai2d-citation" title="展开或收起引用">AI2D</button></td>
        <td>2016 年</td>
        <td>小学科学示意图的视觉问答基准，<b>采用四选一选择题</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="ai2d-examples" title="展开或收起AI2D样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：5.0K<br>问题：15K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="ai2d-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Kembhavi et al. <em>A Diagram Is Worth A Dozen Images.</em> CVPR, 2016. arXiv:1603.07396.
          <a href="https://arxiv.org/abs/1603.07396">论文链接</a>。
          <p><b>任务形式：</b>输入科学示意图与自然语言问题，输出四个候选选项中的正确答案。主要考察图表元素识别、实体关系理解与科学知识推理。</p>
          <p><b>数据规模：</b>包含 5,000+ 张科学示意图和 15,000+ 道问答题；其中训练集 4,000 张图像，测试集 1,000 张图像。</p>
        </td>
      </tr>
      <tr id="ai2d-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/ai2d-example.png' | relative_url }}" alt="AI2D 样例截图，展示青蛙生命周期示意图及模型选择答案" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="textvqa-citation" title="展开或收起引用">TextVQA</button></td>
        <td>2019 年</td>
        <td>自然场景中包含文字的图像问答基准，<b>回答开放式问题</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="textvqa-examples" title="展开或收起TextVQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：3.3K<br>问题：5.7K</td>
        <td class="benchmark-metric-cell">VQA Accuracy</td>
      </tr>
      <tr id="textvqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Singh et al. <em>Towards VQA Models That Can Read.</em> CVPR, 2019.
          <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf">论文链接</a> ·
          <a href="https://textvqa.org/">数据集主页</a>。
          <p>规模按 TextVQA 0.5.1 测试集统计。</p>
        </td>
      </tr>
      <tr id="textvqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/textvqa-example.png' | relative_url }}" alt="TextVQA 样例截图，询问彩色蜡笔的品牌，答案为 Crayola" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mtvqa-citation" title="展开或收起引用">MTVQA</button></td>
        <td>2024 年</td>
        <td><b>多语言</b>视觉问答基准，涵盖文档和自然场景中的<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mtvqa-image-types" title="展开或收起文本密集型图像类型">文本密集型图像</button>，覆盖<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mtvqa-languages" title="展开或收起9种语言">9 种语言</button>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mtvqa-examples" title="展开或收起MTVQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：2.1K<br>问题：6.8K</td>
        <td class="benchmark-metric-cell">Accuracy / ANLS</td>
      </tr>
      <tr id="mtvqa-image-types" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <ul class="benchmark-dataset-list" aria-label="MTVQA 涵盖的文本密集型图像">
            <li>菜单</li>
            <li>标识</li>
            <li>地图</li>
            <li>账单</li>
            <li>PPT</li>
            <li>研究论文</li>
          </ul>
        </td>
      </tr>
      <tr id="mtvqa-languages" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <ul class="benchmark-dataset-list" aria-label="MTVQA 支持的9种语言">
            <li>阿拉伯语</li>
            <li>韩语</li>
            <li>日语</li>
            <li>泰语</li>
            <li>越南语</li>
            <li>俄语</li>
            <li>法语</li>
            <li>德语</li>
            <li>意大利语</li>
          </ul>
        </td>
      </tr>
      <tr id="mtvqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Tang et al. <em>MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering.</em> arXiv:2405.11985, 2024.
          <a href="https://arxiv.org/abs/2405.11985">论文链接</a>。
        </td>
      </tr>
      <tr id="mtvqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mtvqa-example.png' | relative_url }}" alt="MTVQA 样例截图，展示阿拉伯语、俄语、泰语和越南语的文本密集图像问答" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ocr-vqa-citation" title="展开或收起引用">OCR-VQA</button></td>
        <td>2019 年</td>
        <td>书籍封面的<b>文字驱动视觉问答</b>基准，<b>回答开放式问题</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="ocr-vqa-examples" title="展开或收起OCR-VQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：21K<br>问题：100K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="ocr-vqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Mishra et al. <em>OCR-VQA: Visual Question Answering by Reading Text in Images.</em> ICDAR, 2019.
          <a href="https://ieeexplore.ieee.org/document/8978122">论文链接</a> ·
          <a href="https://ocr-vqa.github.io/">数据集主页</a>。
          <p><b>数据集构成：</b>OCR-VQA-200K 包含 207,572 张书籍封面图像和 1,002,146 个问答对，平均每张图像对应 4.83 个问题；包含 117,378 位作者、203,471 个书名、32 种图书类别和 320,794 个不同答案。Train / Val / Test 按 80% / 10% / 10% 划分，测试集约 100K 个问答对。</p>
          <p><b>问题类型：</b>Book Title（书名）、Author Name（作者）、Book Genre（类别）、Year（出版年份）和 Edition（版本）。书名、作者和类别是主要任务；年份与版本仅在部分封面中出现。问题基于模板生成，并经人工改写以增加表达多样性。</p>
          <p><b>评测指标：</b>Accuracy，即预测正确的问题占比。论文报告的最佳基线模型 BLOCK+CNN+W2V 整体准确率为 48.3%；二分类问题为 58.2%，书籍类别识别为 22.0%，作者识别为 42.9%。</p>
        </td>
      </tr>
      <tr id="ocr-vqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/ocr-vqa-example.png' | relative_url }}" alt="OCR-VQA 样例截图，识别书籍封面并回答书名、作者和类别" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ocrbench-citation" title="展开或收起引用">OCRBench</button></td>
        <td>2023 年</td>
        <td>综合 OCR 评测基准，涵盖 5 大任务：文字识别、场景文字问答、文档问答、关键信息抽取和手写数学公式识别。</td>
        <td class="benchmark-count-cell">样本：1.0K</td>
        <td class="benchmark-metric-cell">Score（满分 1,000）</td>
      </tr>
      <tr id="ocrbench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Liu et al. <em>OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models.</em> arXiv:2305.07895, 2023.
          <a href="https://arxiv.org/abs/2305.07895">论文链接</a> ·
          <a href="https://github.com/Yuliang-Liu/MultimodalOCR">官方仓库</a>。
          <p><b>数据集构成：</b>共 1,000 个经人工筛选和校正的问答样本：文字识别 300 个、场景文字问答 200 个、文档问答 200 个、关键信息抽取 200 个、手写数学公式识别 100 个。文字识别覆盖常规、不规则、艺术字、手写、数字串和非语义文本；非语义文本通过打乱单词中字母顺序构造，以检验模型是否真正识别字符。</p>
          <p><b>评测指标：</b>采用统一的 GT-in-output 判定，模型回答包含 Ground Truth 即算正确；五大任务的正确样本数相加得到 Score，满分 1,000。考虑到模型回答常包含解释，论文未直接沿用原始任务的 Exact Match 或 ANLS；同时过滤短答案样本，并人工核验、修正答案及补充多答案。关键信息抽取任务还使用专门提示词，要求模型直接依据图中文字回答。</p>
          <p><b>评测特点：</b>重点考察非语义字符、模糊图像、多语言文字、手写文本及数学公式。论文报告多模态大模型在非语义文本上的平均表现较语义文本下降 57.0%，传统场景文字识别 SOTA 方法约下降 4.6%；模型在手写文本、中文识别和手写数学公式方面仍有不足，文档问答和关键信息抽取也对图像分辨率要求较高。</p>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="docvqa-citation" title="展开或收起引用">DocVQA</button></td>
        <td>2021 年</td>
        <td>真实文档图像的<b>文档视觉问答</b>基准，图像来自<b>工业文献档案库</b>，覆盖<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="docvqa-industries" title="展开或收起5个行业">5 个行业</button>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="docvqa-examples" title="展开或收起DocVQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：1.3K<br>问题：5.2K</td>
        <td class="benchmark-metric-cell">ANLS / Accuracy</td>
      </tr>
      <tr id="docvqa-industries" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <p>工业文献档案库覆盖以下 5 个行业：</p>
          <ul class="benchmark-dataset-list">
            <li>Tobacco：烟草</li>
            <li>Food：食品</li>
            <li>Drug：药品</li>
            <li>Fossil Fuel：化石燃料</li>
            <li>Chemical：化工</li>
          </ul>
        </td>
      </tr>
      <tr id="docvqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Mathew et al. <em>DocVQA: A Dataset for VQA on Document Images.</em> WACV, 2021.
          <a href="https://arxiv.org/abs/2007.00398">论文链接</a>。
          <p><b>数据规模：</b>测试集 5,188 个问题、1,287 张图像；完整数据集 50,000 个问题、12,767 张图像。数据划分：Train / Val / Test = 80% / 10% / 10%。</p>
          <p><b>ANLS（Average Normalized Levenshtein Similarity）：</b>基于编辑距离计算预测答案与标准答案的相似度，降低 OCR 拼写错误等细微差异带来的惩罚。</p>
          <p><b>Accuracy：</b>预测答案与任意一个 Ground Truth 完全匹配的比例。</p>
        </td>
      </tr>
      <tr id="docvqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/docvqa-example.png' | relative_url }}" alt="DocVQA 样例截图，询问信件上的邮政编码、日期和公司地址" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="chartqa-citation" title="展开或收起引用">ChartQA</button></td>
        <td>2022 年</td>
        <td>真实世界图表的视觉问答基准，<b>回答开放式问题</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="chartqa-examples" title="展开或收起ChartQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：1.6K<br>问题：2.5K</td>
        <td class="benchmark-metric-cell">Relaxed Accuracy</td>
      </tr>
      <tr id="chartqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Masry et al. <em>ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning.</em> Findings of ACL, 2022.
          <a href="https://arxiv.org/abs/2203.10244">论文链接</a>。
          <p><b>数据集构成：</b>ChartQA-H 包含 9,608 个由人工编写的问答对；ChartQA-M 包含 23,111 个基于人工撰写图表摘要、由 T5 自动生成的问答对。全数据集共 20,882 张真实世界图表，来源包括 Statista、Pew Research、Our World in Data（OWID）和 OECD。</p>
          <p><b>问题类型：</b>Data Retrieval（13.0%）、Visual（10.7%）、Compositional（43.0%）、Visual &amp; Compositional（33.3%）。</p>
          <p><b>Relaxed Accuracy：</b>数值答案与标准答案的误差在 5% 以内即视为正确；非数值答案采用 Exact Match。</p>
          <p>ChartQA 不仅要求读取图表文本或数值，也考察结合视觉属性与底层数据进行组合推理，例如比较数据差异、计算平均值或寻找最大值。</p>
        </td>
      </tr>
      <tr id="chartqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/chartqa-example.png' | relative_url }}" alt="ChartQA 样例截图，询问巴西经济评价意见差异最大的年份及橙色折线峰值" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="infographic-vqa-citation" title="展开或收起引用">InfoVQA</button></td>
        <td>2022 年</td>
        <td>面向<b>信息图（Infographics）</b>的视觉问答基准，输入包含文本、图表、表格、地图等视觉元素的信息图像。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="infographic-vqa-examples" title="展开或收起InfoVQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：0.6K<br>问题：3.3K</td>
        <td class="benchmark-metric-cell">ANLS / Accuracy</td>
      </tr>
      <tr id="infographic-vqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Mathew et al. <em>InfographicVQA.</em> WACV, 2022.
          <a href="https://openaccess.thecvf.com/content/WACV2022/html/Mathew_InfographicVQA_WACV_2022_paper.html">论文链接</a>。
        </td>
      </tr>
      <tr id="infographic-vqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/infovqa-example.png' | relative_url }}" alt="InfoVQA 样例截图，展示加拿大邮政与亚马逊配送员工人数和媒体覆盖地图，以及三个基于信息图的问答" loading="lazy">
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 5. 高分辨率视觉理解

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="高分辨率视觉理解基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mme-realworld-citation" title="展开或收起引用">MME-RealWorld</button></td>
        <td>2025 年</td>
        <td>面向真实世界复杂场景的<b>高分辨率多模态理解</b>基准，重点考察细粒度视觉感知与复杂场景推理，覆盖<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mme-realworld-domains" title="展开或收起5大领域与评测任务">5 大领域与 43 个子任务</button>。图像平均分辨率约为 <b>2000×1500（接近 2K）</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mme-realworld-examples" title="展开或收起MME-RealWorld样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：13.4K<br>问题：29.4K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="mme-realworld-domains" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table mme-realworld-domains-table" aria-label="MME-RealWorld 的五大领域与评测任务">
            <thead>
              <tr><th scope="col">领域</th><th scope="col">Perception（感知）</th><th scope="col">Reasoning（推理）</th></tr>
            </thead>
            <tbody>
              <tr><td>OCR in the Wild（真实场景文字识别）</td><td>联系信息、身份信息、商品广告、标识文字、书籍地图及海报识别</td><td>场景理解、人物角色理解</td></tr>
              <tr><td>Remote Sensing（遥感）</td><td>物体计数、颜色识别、空间关系理解</td><td>—</td></tr>
              <tr><td>Diagram and Table（图表与表格）</td><td>表格信息提取、图表信息提取</td><td>表格推理、图表推理</td></tr>
              <tr><td>Autonomous Driving（自动驾驶）</td><td>物体识别、属性识别、物体计数</td><td>意图预测、交互关系理解、驾驶员注意力理解</td></tr>
              <tr><td>Monitoring（视频监控）</td><td>物体计数、位置识别、属性识别</td><td>多物体数量计算、意图推理、属性推理</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="mme-realworld-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Zhang et al. <em>MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?</em> ICLR, 2025. arXiv:2408.13257.
          <a href="https://arxiv.org/abs/2408.13257">论文链接</a>。
          <p><b>数据构建：</b>从超过 300K 张公开数据集及互联网图像中筛选出 13,366 张高质量图像，由 25 名专业标注人员和 7 名多模态模型领域专家参与构建与审核。每题为五选一，选项 E 表示图像中不存在相关对象或信息。</p>
          <p><b>评测指标：</b>Accuracy。Avg 按各子任务样本数加权平均；Avg-C 为各子任务准确率的非加权平均。论文另提供中文版本 MME-RealWorld-CN，包含 5,917 个问答对。</p>
        </td>
      </tr>
      <tr id="mme-realworld-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/mme-realworld-example.png' | relative_url }}" alt="MME-RealWorld 样例截图，展示真实场景文字识别与遥感图像问答" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="vstar-bench-citation" title="展开或收起引用">V* Bench</button></td>
        <td>2024 年</td>
        <td>面向约 2K 高分辨率图像的<b>细粒度视觉搜索与定位</b>基准，包含属性识别和空间关系推理两类任务，要求定位目标并识别其属性或与其他物体的相对位置。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="vstar-bench-examples" title="展开或收起 V* Bench 样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：0.2K<br>问题：0.2K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="vstar-bench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Wu and Xie. <em>V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs.</em> CVPR, 2024.
          <p><b>数据与任务：</b>从 SA-1B 筛选 191 张高分辨率图像，平均分辨率约 2K。Attribute Recognition 共 115 道，识别指定物体的颜色、材质等属性；Spatial Relationship Reasoning 共 76 道，判断两个物体的相对空间位置。</p>
          <p><b>评测方式：</b>多项选择题；开放式问题设 4 个选项，二分类问题设 2 个选项，以 Accuracy 评估。</p>
        </td>
      </tr>
      <tr id="vstar-bench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/vstar-bench-example.png' | relative_url }}" alt="V* Bench 样例：在玩具堆的高分辨率图像中定位猩猩，并判断它拿着什么乐器" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="hr-bench-citation" title="展开或收起引用">HR-Bench</button></td>
        <td>2025 年</td>
        <td>评估多模态大模型在<b>超高分辨率图像</b>中的细粒度感知能力，提供 4K 和 8K 设置，包含单实例细粒度感知（FSP）与跨实例细粒度感知（FCP）。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="hr-bench-examples" title="展开或收起 HR-Bench 样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：0.2K<br>问题：0.2K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="hr-bench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Wang et al. <em>Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models.</em> AAAI, 2025.
          <p><b>数据与任务：</b>200 张来自 DIV8K 和互联网的 8K 图像，人工标注问题与答案。FSP 共 100 道，识别单个物体的颜色、材质等属性；FCP 共 100 道，判断多个物体间的相对位置。另提供 HR-Bench 4K 版本，通过标注目标位置并裁剪原始 8K 图像生成。</p>
          <p><b>评测方式：</b>多项选择题采用 Cyclic Permutation 循环移位选项顺序，以降低选项顺序对模型表现的影响，最终以 Accuracy 衡量性能。</p>
        </td>
      </tr>
      <tr id="hr-bench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/hr-bench-example.png' | relative_url }}" alt="HR-Bench 样例：高分辨率迪士尼城堡场景中，放大定位雨伞并询问其颜色" loading="lazy">
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 6. 传统视觉问答与图像描述生成

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="传统视觉问答与图像描述生成基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="vqav2-citation" title="展开或收起引用">VQAv2</button></td>
        <td>2017 年</td>
        <td>通用开放式视觉问答基准，给定图像和自然语言问题，要求模型结合视觉信息、语言理解与常识知识生成答案。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="vqav2-examples" title="展开或收起VQAv2样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：81K<br>问题：448K</td>
        <td class="benchmark-metric-cell">VQA Accuracy</td>
      </tr>
      <tr id="vqav2-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Goyal et al. <em>Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering.</em> CVPR, 2017.
          <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_V_in_CVPR_2017_paper.pdf">论文链接</a> ·
          <a href="https://visualqa.org/download.html">数据集主页</a>。
          <p><b>数据集构成：</b>VQAv2.0 完整数据集包含 204,721 张 COCO 图像、1,105,904 个问题和 11,059,040 个人工标注答案。训练集包含 443,757 个问题，验证集包含 214,354 个问题，测试集包含 447,793 个问题。</p>
        </td>
      </tr>
      <tr id="vqav2-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/vqav2-example.png' | relative_url }}" alt="VQAv2 样例截图，展示关于戴眼镜的人、孩子位置、雨伞方向和床上儿童数量的问答" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="vizwiz-citation" title="展开或收起引用">VizWiz</button></td>
        <td>2018 年</td>
        <td>面向视障人士的真实场景视觉问答基准，由视障用户使用手机拍摄图片并口述问题。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="vizwiz-examples" title="展开或收起VizWiz样例">[样例]</button></td>
        <td class="benchmark-count-cell">问题：31K</td>
        <td class="benchmark-metric-cell">VQA Accuracy 等 </td>
      </tr>
      <tr id="vizwiz-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Gurari et al. <em>VizWiz Grand Challenge: Answering Visual Questions from Blind People.</em> CVPR, 2018.
          <a href="https://arxiv.org/abs/1802.08218">论文链接</a>。
        </td>
      </tr>
      <tr id="vizwiz-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/vizwiz-example.png' | relative_url }}" alt="VizWiz 样例截图，展示视障用户询问化妆品是否含防晒成分以及识别十欧元纸币" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="gqa-citation" title="展开或收起引用">GQA</button></td>
        <td>2019 年</td>
        <td>基于 Visual Genome 场景图自动生成的<b>组合式视觉问答</b>数据集，通过遍历物体、属性及关系构造多样化问题。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="gqa-examples" title="展开或收起GQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：113K<br>问题：23M</td>
        <td class="benchmark-metric-cell">Accuracy </td>
      </tr>
      <tr id="gqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Hudson et al. <em>GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.</em> CVPR, 2019.
          <a href="https://arxiv.org/abs/1902.09506">论文链接</a>。
        </td>
      </tr>
      <tr id="gqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/gqa-example.png' | relative_url }}" alt="GQA 样例截图，展示两张场景图及关于物体属性、位置、颜色和关系的组合式问题" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="okvqa-citation" title="展开或收起引用">OK-VQA</button></td>
        <td>2019 年</td>
        <td>面向<b>外部知识(Outside Knowledge VQA)</b>的视觉问答基准，要求模型结合图像内容与外部世界知识回答问题。涵盖 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="okvqa-categories" title="展开或收起10类知识领域">10 类知识领域</button>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="okvqa-examples" title="展开或收起OK-VQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：14K<br>问题：14K</td>
        <td class="benchmark-metric-cell">VQA Accuracy</td>
      </tr>
      <tr id="okvqa-categories" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <table class="benchmark-subtask-table okvqa-categories-table" aria-label="OK-VQA 的 10 类知识领域">
            <caption>OK-VQA 的 10 类知识领域</caption>
            <thead>
              <tr>
                <th scope="col">序号</th>
                <th scope="col">英文类别</th>
                <th scope="col">中文含义</th>
              </tr>
            </thead>
            <tbody>
              <tr><th scope="row">1</th><td>Vehicles and Transportation</td><td>交通工具与运输</td></tr>
              <tr><th scope="row">2</th><td>Brands, Companies and Products</td><td>品牌、公司与产品</td></tr>
              <tr><th scope="row">3</th><td>Objects, Materials and Clothing</td><td>物体、材料与服装</td></tr>
              <tr><th scope="row">4</th><td>Sports and Recreation</td><td>体育与休闲活动</td></tr>
              <tr><th scope="row">5</th><td>Cooking and Food</td><td>烹饪与食物</td></tr>
              <tr><th scope="row">6</th><td>Geography, History, Language and Culture</td><td>地理、历史、语言与文化</td></tr>
              <tr><th scope="row">7</th><td>People and Everyday Life</td><td>人物与日常生活</td></tr>
              <tr><th scope="row">8</th><td>Plants and Animals</td><td>植物与动物</td></tr>
              <tr><th scope="row">9</th><td>Science and Technology</td><td>科学与技术</td></tr>
              <tr><th scope="row">10</th><td>Weather and Climate</td><td>天气与气候</td></tr>
            </tbody>
          </table>
        </td>
      </tr>
      <tr id="okvqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Marino et al. <em>OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge.</em> CVPR, 2019.
          <a href="https://arxiv.org/abs/1906.00067">arXiv:1906.00067</a>。
        </td>
      </tr>
      <tr id="okvqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/okvqa-example.png' | relative_url }}" alt="OK-VQA 样例截图，询问图中教堂最常在星期几开放，答案为 Sunday" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="flickr30k-citation" title="展开或收起引用">Flickr30K</button></td>
        <td>2014 年</td>
        <td>经典<b>图像描述生成（Image Captioning）</b>基准，包含日常生活、人物活动和自然场景等图像。每张图像由 5 名不同标注者独立撰写英文描述。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="flickr30k-examples" title="展开或收起Flickr30K样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：32K<br>描述：159K</td>
        <td class="benchmark-metric-cell"> CIDEr等</td>
      </tr>
      <tr id="flickr30k-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Young et al. <em>From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.</em> TACL, 2014.
          <a href="https://aclanthology.org/Q14-1006/">论文链接</a>。
        </td>
      </tr>
      <tr id="flickr30k-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/flickr30k-example.png' | relative_url }}" alt="Flickr30K 样例截图，展示两张日常场景图片及每张图片对应的五条英文描述" loading="lazy">
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 7. 视觉定位与计数

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="视觉定位与计数基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 54%">
      <col style="width: 12%">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="countbench-citation" title="展开或收起引用">CountBench</button></td>
        <td>2023 年（ICCV）</td>
        <td>面向真实场景图像的<b>物体计数</b>基准，评估模型根据文本描述判断图像中指定类别的物体数量，数量范围为 2–10。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="countbench-examples" title="展开或收起 CountBench 样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：0.5K</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="countbench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          <em>Teaching CLIP to Count to Ten.</em> ICCV, 2023.
          <p><b>数据集构成：</b>从 LAION-400M 图文数据集中自动筛选，并经人工验证，共包含 540 张图像及对应文本描述。物体数量从 2 到 10，每种数量各有 60 张图像，确保指定物体数量与文本描述一致。</p>
          <p><b>评测方式：</b>采用 Zero-shot Counting Classification。将每张图像的原始描述分别替换为数量 2–10，生成 9 个候选描述；计算图像与各描述的相似度，以最高相似度对应的数量作为预测，并以 Accuracy 评估。</p>
        </td>
      </tr>
      <tr id="countbench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/countbench-example.png' | relative_url }}" alt="CountBench 样例截图，展示从两只斑马到十朵圣诞花的不同数量图像与文本描述" loading="lazy">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="refcoco-citation" title="展开或收起引用">RefCOCO</button></td>
        <td>2016 年</td>
        <td>基于 MS COCO 的<b>视觉指代表达理解</b>基准，给定图像和自然语言描述，要求定位对应的目标物体；允许使用空间位置描述。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="refcoco-examples" title="展开或收起RefCOCO样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：20K<br>描述：142K</td>
        <td class="benchmark-metric-cell">Acc@0.5</td>
      </tr>
      <tr id="refcoco-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Yu et al. <em>Modeling Context in Referring Expressions.</em> ECCV, 2016.
          <a href="https://arxiv.org/abs/1608.00272">论文链接</a>。
          <p><b>任务形式：</b>输入图像和自然语言指代表达（Referring Expression），输出目标物体的 Bounding Box，考察视觉属性识别、空间关系理解与细粒度定位。</p>
          <p><b>数据划分：</b>RefCOCO 和 RefCOCO+ 使用 Train / Val / TestA / TestB 划分；TestA 含多个人物实例的图像，TestB 含多个非人物物体实例的图像。</p>
          <p><b>Acc@0.5：</b>预测框与 Ground Truth 的 IoU 大于 0.5 时计为正确，统计定位准确率。</p>
        </td>
      </tr>
      <tr id="refcoco-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/refcoco-series-example.png' | relative_url }}" alt="RefCOCO、RefCOCO+ 和 RefCOCOg 样例截图，展示长颈鹿目标及不同指代表达">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="refcocoplus-citation" title="展开或收起引用">RefCOCO+</button></td>
        <td>2016 年</td>
        <td>在 RefCOCO 基础上禁止使用 left、right 等<b>绝对位置词</b>，要求模型更多依赖物体外观属性进行目标定位。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="refcocoplus-examples" title="展开或收起RefCOCO+样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：20K<br>描述：142K</td>
        <td class="benchmark-metric-cell">Acc@0.5</td>
      </tr>
      <tr id="refcocoplus-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Yu et al. <em>Modeling Context in Referring Expressions.</em> ECCV, 2016.
          <a href="https://arxiv.org/abs/1608.00272">论文链接</a>。
          <p><b>数据划分与指标：</b>采用 Train / Val / TestA / TestB 划分，TestA 含多个人物实例图像，TestB 含多个非人物物体实例图像。Acc@0.5 以预测框与 Ground Truth 的 IoU 是否大于 0.5 判断定位正确。</p>
        </td>
      </tr>
      <tr id="refcocoplus-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/refcoco-series-example.png' | relative_url }}" alt="RefCOCO、RefCOCO+ 和 RefCOCOg 样例截图，展示长颈鹿目标及不同指代表达">
          </div>
        </td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="refcocog-citation" title="展开或收起引用">RefCOCOg</button></td>
        <td>2016 年</td>
        <td>基于 MS COCO 的视觉指代表达理解基准，采用非交互式众包标注，表达通常更长、更复杂，强调<b>细粒度语义理解</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="refcocog-examples" title="展开或收起RefCOCOg样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：26K<br>描述：95K</td>
        <td class="benchmark-metric-cell">Acc@0.5</td>
      </tr>
      <tr id="refcocog-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Mao et al. <em>Generation and Comprehension of Unambiguous Object Descriptions.</em> CVPR, 2016.
          <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Mao_Generation_and_Comprehension_CVPR_2016_paper.html">论文链接</a>。
          <p><b>数据划分：</b>提供 Google 和 UMD 两种划分，分别按物体实例和图像划分。Acc@0.5 以预测框与 Ground Truth 的 IoU 是否大于 0.5 判断定位正确。</p>
        </td>
      </tr>
      <tr id="refcocog-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          <div class="benchmark-examples-panel">
            <img class="benchmark-example-image" src="{{ '/assets/img/benchmarks/refcoco-series-example.png' | relative_url }}" alt="RefCOCO、RefCOCO+ 和 RefCOCOg 样例截图，展示长颈鹿目标及不同指代表达">
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
