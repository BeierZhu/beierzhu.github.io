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
  .benchmark-citation-toggle, .benchmark-datasets-toggle, .benchmark-performance-toggle, .benchmark-examples-toggle { padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; font-weight: normal; text-decoration: none; cursor: pointer; }
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
        <td class="benchmark-count-cell">图像：1,187 <br>问题：2,374 </td>
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
    </tbody>
  </table>
</div>

<br>
## 2. 文本与图表理解

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="文本、图表与示意图理解基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 48%">
      <col style="width: 18%">
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
        <td class="benchmark-count-cell">图像：5,000+<br>问题：15,000+</td>
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
        <td class="benchmark-count-cell">图像：3,289 <br>问题：5,734</td>
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ocr-vqa-citation" title="展开或收起引用">OCR-VQA</button></td>
        <td>2019 年</td>
        <td>书籍封面的<b>文字驱动视觉问答</b>基准，<b>回答开放式问题</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="ocr-vqa-examples" title="展开或收起OCR-VQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：约 20.8K<br>问题： 100K</td>
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="docvqa-citation" title="展开或收起引用">DocVQA</button></td>
        <td>2021 年</td>
        <td>真实文档图像的<b>文档视觉问答</b>基准，图像来自<b>工业文献档案库</b>，覆盖<button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="docvqa-industries" title="展开或收起5个行业">5 个行业</button>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="docvqa-examples" title="展开或收起DocVQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：1,287 <br>问题：5,188</td>
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
        <td class="benchmark-count-cell">图像： 1.6K <br>问题：2,500</td>
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
    </tbody>
  </table>
</div>

<br>
## 3. 传统视觉问答

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="传统视觉问答基准">
    <colgroup>
      <col style="width: 12%">
      <col style="width: 10%">
      <col style="width: 48%">
      <col style="width: 18%">
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
        <td class="benchmark-count-cell">图像：81,434<br>问题：447,793</td>
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="okvqa-citation" title="展开或收起引用">OK-VQA</button></td>
        <td>2019 年</td>
        <td>面向<b>外部知识(Outside Knowledge VQA)</b>的视觉问答基准，要求模型结合图像内容与外部世界知识回答问题。涵盖 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="okvqa-categories" title="展开或收起10类知识领域">10 类知识领域</button>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="okvqa-examples" title="展开或收起OK-VQA样例">[样例]</button></td>
        <td class="benchmark-count-cell">图像：14,031<br>问题：14,055</td>
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
    </tbody>
  </table>
</div>

<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
