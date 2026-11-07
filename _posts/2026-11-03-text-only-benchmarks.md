---
layout: post
title: "大模型测评基准I：纯文本基准"
date: 2026-11-03
permalink: /blog/2026/text-only-benchmarks/
description: "本文旨在系统梳理主流基准，便于个人查阅参考。"
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
  .benchmark-example { padding: .75rem 0; border-bottom: 1px solid var(--global-divider-color); }
  .benchmark-example:first-child { padding-top: 0; }
  .benchmark-example:last-child { padding-bottom: 0; border-bottom: 0; }
  .benchmark-example p { margin-bottom: .4rem; }
  .benchmark-example p:last-child { margin-bottom: 0; }
  .benchmark-preview-table b, .benchmark-examples-panel b { font-weight: bold; }
  .benchmark-example .highlight { margin: 0; max-width: 100%; }
  .benchmark-example pre { max-width: 100%; overflow-x: auto; white-space: pre; }
  .benchmark-example-correct { background-color: rgba(97, 169, 189, .25); }
  .benchmark-example-result-correct { color: #198754; font-weight: bold; }
  .benchmark-example-result-incorrect { color: #bd4147; font-weight: bold; }
  .benchmark-winogrande-example { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: .5rem 1rem; }
  .benchmark-winogrande-example p { margin-bottom: 0; }
  .benchmark-winogrande-choices { white-space: nowrap; }
  @media (max-width: 575px) { .benchmark-winogrande-example { grid-template-columns: minmax(0, 1fr); } }
---


## 1. 通用推理

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table>
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
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="pile-citation" title="展开或收起引用">Pile-test</button>
        </td>
        <td>2020 年</td>
        <td>The Pile 的独立测试集，覆盖 <b><button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="pile-datasets" title="展开或收起22个子数据集名称">22</button> 个多领域子数据集</b>，涵盖学术论文、书籍、代码、网页、法律文档及在线对话等，主要用于评估语言模型的<b>跨领域语言建模能力</b>。</td>
        <td>214,584</td>
        <td>PPL / BPB</td>
      </tr>
      <tr id="pile-datasets" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          <ul class="benchmark-dataset-list">
            <li>Pile-CC</li>
            <li>PubMed Central</li>
            <li>Books3</li>
            <li>OpenWebText2</li>
            <li>ArXiv</li>
            <li>Github</li>
            <li>FreeLaw</li>
            <li>Stack Exchange</li>
            <li>USPTO Backgrounds</li>
            <li>PubMed Abstracts</li>
            <li>Gutenberg (PG-19)</li>
            <li>OpenSubtitles</li>
            <li>Wikipedia (en)</li>
            <li>DM Mathematics</li>
            <li>Ubuntu IRC</li>
            <li>BookCorpus2</li>
            <li>EuroParl</li>
            <li>HackerNews</li>
            <li>YoutubeSubtitles</li>
            <li>PhilPapers</li>
            <li>NIH ExPorter</li>
            <li>Enron Emails</li>
          </ul>
        </td>
      </tr>
      <tr id="pile-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Gao et al. <em>The Pile: An 800GB Dataset of Diverse Text for Language Modeling.</em> arXiv:2101.00027. <a href="https://arxiv.org/abs/2101.00027">论文链接</a> · <a href="https://huggingface.co/datasets/EleutherAI/pile_val_test/viewer/default/test">测试集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="natural-questions-citation" title="展开或收起引用">Natural Questions</button>
        </td>
        <td>2019 年</td>
        <td>问题来自<b>真实 Google 搜索查询</b>，要求模型结合 Wikipedia 文档定位长答案并提取短答案，同时判断文档中是否存在答案，评估模型的<b>长文档理解、答案定位与问答能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="natural-questions-examples" title="展开或收起三个文字样例">[样例]</button></td>
        <td>7,842</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="natural-questions-performance" title="展开或收起 Qwen 系列模型的 EM 柱状图">EM</button></td>
      </tr>
      <tr id="natural-questions-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/natural-questions-examples.liquid %}
        </td>
      </tr>
      <tr id="natural-questions-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/natural-questions-qwen.liquid %}
        </td>
      </tr>
      <tr id="natural-questions-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Kwiatkowski et al. <em>Natural Questions: A Benchmark for Question Answering Research.</em> TACL, 2019. <a href="https://aclanthology.org/Q19-1026/">论文链接</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="triviaqa-citation" title="展开或收起引用">TriviaQA</button>
        </td>
        <td>2017 年</td>
        <td>问题由<b>知识问答爱好者编写</b>，配有从 Wikipedia 和网页收集的证据文档。问题与证据之间常存在词汇差异，需要整合多句话定位答案，评估模型的<b>事实知识、阅读理解与跨句推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="triviaqa-examples" title="展开或收起文字样例">[样例]</button></td>
        <td title="Unfiltered 版本">10,832</td>
        <td>EM / F1</td>
      </tr>
      <tr id="triviaqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/triviaqa-examples.liquid %}
        </td>
      </tr>
      <tr id="triviaqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Joshi et al. <em>TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.</em> ACL, 2017. <a href="https://aclanthology.org/P17-1147/">论文链接</a> · <a href="https://nlp.cs.washington.edu/triviaqa/">数据集主页</a>。规模采用 <a href="https://huggingface.co/datasets/mandarjoshi/trivia_qa/viewer/unfiltered/test">Unfiltered 版本</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="piqa-citation" title="展开或收起引用">PIQA</button>
        </td>
        <td>2019 年</td>
        <td>围绕<b>日常生活中的物理常识与操作任务</b>，给定一个目标和两个候选方案，要求选出更合理、可行的方案。评估模型对<b>物体属性、工具用途及物理交互</b>的理解与常识推理能力。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="piqa-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>3,084</td>
        <td>Accuracy</td>
      </tr>
      <tr id="piqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/piqa-examples.liquid %}
        </td>
      </tr>
      <tr id="piqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Bisk et al. <em>PIQA: Reasoning about Physical Commonsense in Natural Language.</em> AAAI, 2020（预印本于 2019 年发布）。<a href="https://arxiv.org/abs/1911.11641">论文链接</a> · <a href="https://yonatanbisk.com/piqa/">数据集主页</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="winogrande-citation" title="展开或收起引用">WinoGrande</button>
        </td>
        <td>2019 年</td>
        <td>以<b>指代消解</b>（判断代词
          如 it、she 指代句中的哪个人或物）为核心，给定句子与两个候选对象，要求根据语境补全空缺、确定正确指代。采用<b>对抗过滤</b>减少词语关联等数据偏差，评估模型的<b>常识推理与语境理解能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="winogrande-examples" title="展开或收起两个文字样例">[样例]</button></td>
        <td>1,767</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="winogrande-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="winogrande-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/winogrande-examples.liquid %}
        </td>
      </tr>
      <tr id="winogrande-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/winogrande-qwen.liquid %}
        </td>
      </tr>
      <tr id="winogrande-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Sakaguchi et al. <em>WinoGrande: An Adversarial Winograd Schema Challenge at Scale.</em> AAAI, 2020（预印本于 2019 年发布）。<a href="https://arxiv.org/abs/1907.10641">论文链接</a> · <a href="https://huggingface.co/datasets/allenai/winogrande/viewer/winogrande_xl/test">数据集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="hellaswag-citation" title="展开或收起引用">HellaSwag</button>
        </td>
        <td>2019 年</td>
        <td>围绕<b>日常活动与事件续写</b>，给定一段情境描述，从四个候选续写中选择最合理的后续情节。干扰选项通过<b>对抗过滤</b>筛选，语法自然但语义不合常理，评估模型的<b>事件预测、情境理解与常识推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="hellaswag-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>10,003</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="hellaswag-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="hellaswag-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/hellaswag-examples.liquid %}
        </td>
      </tr>
      <tr id="hellaswag-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/hellaswag-qwen.liquid %}
        </td>
      </tr>
      <tr id="hellaswag-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Zellers et al. <em>HellaSwag: Can a Machine Really Finish Your Sentence?</em> ACL, 2019. <a href="https://arxiv.org/abs/1905.07830">论文链接</a> · <a href="https://rowanzellers.com/hellaswag/">数据集主页</a> · <a href="https://huggingface.co/datasets/Rowan/hellaswag/viewer/default/test">数据集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="big-bench-citation" title="展开或收起引用">BIG-Bench</button>
        </td>
        <td>2022 年</td>
        <td>由研究社区协作构建的<b>综合能力评测合集</b>，论文收录 <b>204 个任务</b>，覆盖语言理解、数学、常识推理、代码、多语言与社会偏差等领域。包含静态题目与可编程交互任务，用于考察模型的<b>多维能力、泛化表现与规模增长规律</b>。</td>
        <td title="不同子任务提供不同数量的评测样本，部分任务支持程序化生成；没有统一测试集规模。">1-2M</td>
        <td title="各任务使用首选指标（如 Accuracy、EM、BLEU、ROUGE），按该任务设定的低分和高分基准归一化，再跨任务取平均。">归一化均分</td>
      </tr>
      <tr id="big-bench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">BIG-bench authors. <em>Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models.</em> TMLR, 2023（预印本于 2022 年发布）。<a href="https://arxiv.org/abs/2206.04615">论文链接</a> · <a href="https://github.com/google/BIG-bench">官方仓库</a>。规模按各子任务及评测采样设置确定；汇总时将各任务首选指标按其低分、高分基准归一化至 0–100 的参考区间后取平均，实际得分可以超出该区间。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="big-bench-lite-citation" title="展开或收起引用">BIG-Bench Lite</button>
        </td>
        <td>2022 年</td>
        <td>BIG-Bench 的<b>轻量代表性子集</b>，精选 <b><button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="big-bench-lite-tasks" title="展开或收起24个任务名称">24</button> 个 JSON 格式任务</b>，覆盖推理、代码、多语言及偏差测量等能力。不包含需要复杂交互的可编程任务，便于以<b>更低评测成本</b>比较模型的综合表现</td>
        <td title="24 是任务数量，不是测试样本数量；各子任务的样本数和评测采样设置不同。">5K</td>
        <td title="沿用各任务的首选指标，经任务内归一化后对24个任务取平均。">归一化均分</td>
      </tr>
      <tr id="big-bench-lite-tasks" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          {% include benchmarks/big-bench-lite-tasks.liquid %}
        </td>
      </tr>
      <tr id="big-bench-lite-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">BIG-bench authors. <em>Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models.</em> TMLR, 2023（预印本于 2022 年发布），§2.2 / Table 1. <a href="https://arxiv.org/abs/2206.04615">论文链接</a> · <a href="https://github.com/google/BIG-bench#big-bench-lite-leaderboard">官方说明</a>。Lite 与完整 BIG-Bench 共用论文来源，任务数为 24；测试样本量仍需按具体任务和采样设置确定。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="bbh-citation" title="展开或收起引用">BBH</button>
        </td>
        <td>2022 年</td>
        <td>BIG-Bench Hard，从 BIG-Bench 中筛选出 <b><button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="bbh-tasks" title="展开或收起23类任务名称">23</button> 类高难度任务</b>，原研究中的模型尚未超过这些任务的人类平均表现。覆盖逻辑、算术、符号与常识推理，重点考察<b>多步推理能力</b>，常用于研究<b>思维链（CoT）提示</b>的效果。</td>
        <td title="官方27个任务变体数据文件合计的评测样本数，不含few-shot提示样例。">6,511</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="bbh-performance" title="展开或收起 Qwen 系列模型的 EM 柱状图">EM</button></td>
      </tr>
      <tr id="bbh-tasks" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          {% include benchmarks/bbh-tasks.liquid %}
        </td>
      </tr>
      <tr id="bbh-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/bbh-qwen.liquid %}
        </td>
      </tr>
      <tr id="bbh-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Suzgun et al. <em>Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.</em> arXiv:2210.09261, 2022. <a href="https://arxiv.org/abs/2210.09261">论文链接</a> · <a href="https://github.com/suzgunmirac/BIG-Bench-Hard">官方仓库</a>。23 类任务中，逻辑演绎和物体交换追踪各含 3、5、7 个对象的变体，共 27 个数据文件、6,511 条评测样本。指标按生成答案与标准答案的完全匹配（EM）计算正确率，再跨任务汇总。</td>
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
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="arc-citation" title="展开或收起引用">ARC-C</button>
        </td>
        <td>2018 年</td>
        <td>来自<b>中小学科学考试的多项选择题</b>（4 选项 QA）。ARC-Challenge 收录检索和词共现基线均无法答对的题目，重点考察模型的<b>科学知识运用与推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="arc-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>1,172</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="arc-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="arc-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/arc-examples.liquid %}
        </td>
      </tr>
      <tr id="arc-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/arc-qwen.liquid %}
        </td>
      </tr>
      <tr id="arc-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Clark et al. <em>Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.</em> arXiv:1803.05457, 2018. <a href="https://arxiv.org/abs/1803.05457">论文链接</a> · <a href="https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge/test">ARC-C 测试集</a> · <a href="https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy/test">ARC-E 测试集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="theoremqa-citation" title="展开或收起引用">TheoremQA</button>
        </td>
        <td>2023 年</td>
        <td>面向大学阶段的<b>定理驱动型科学问答基准</b>，涵盖数学、物理、电子工程与计算机科学、金融四大领域，包含约 <b>354 个定理和 800 道问题</b>，重点评估模型运用专业定理解决跨学科科学问题的<b>知识与推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="theoremqa-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>800</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="theoremqa-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="theoremqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/theoremqa-examples.liquid %}
        </td>
      </tr>
      <tr id="theoremqa-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/theoremqa-qwen.liquid %}
        </td>
      </tr>
      <tr id="theoremqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Chen et al. <em>TheoremQA: A Theorem-driven Question Answering dataset.</em> EMNLP, 2023. <a href="https://arxiv.org/abs/2305.12524">论文链接</a> · <a href="https://github.com/TIGER-AI-Lab/TheoremQA">官方仓库</a> · <a href="https://huggingface.co/datasets/TIGER-Lab/TheoremQA">数据集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmlu-citation" title="展开或收起引用">MMLU</button>
        </td>
        <td>2020 年</td>
        <td>MMLU（Massive Multitask Language Understanding）是一个涵盖 <b>57 个学科四选一题目</b>的综合知识评测基准，用于评估大语言模型在 STEM、人文、社会科学及专业领域的<b>知识与理解能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mmlu-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>14,042</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="mmlu-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="mmlu-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/mmlu-examples.liquid %}
        </td>
      </tr>
      <tr id="mmlu-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/mmlu-qwen.liquid %}
        </td>
      </tr>
      <tr id="mmlu-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Hendrycks et al. <em>Measuring Massive Multitask Language Understanding.</em> ICLR, 2021（预印本于 2020 年发布）。<a href="https://arxiv.org/abs/2009.03300">论文链接</a> · <a href="https://github.com/hendrycks/test">官方仓库</a> · <a href="https://huggingface.co/datasets/cais/mmlu/viewer/all/test">测试集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmlu-redux-citation" title="展开或收起引用">MMLU-Redux</button>
        </td>
        <td>2024 年</td>
        <td>对 MMLU 进行<b>人工审核与重新标注</b>的修正版，旨在纠正错误答案与歧义题目；抽样发现约 <b>6.49%</b> 的题目存在错误。</td>
        <td title="MMLU-Redux 2.0：57个学科，每科100条测试样本。">5,700</td>
        <td>Accuracy</td>
      </tr>
      <tr id="mmlu-redux-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Gema et al. <em>Are We Done with MMLU?</em> NAACL-HLT, 2025（预印本于 2024 年发布）。<a href="https://arxiv.org/abs/2406.04127">论文链接</a> · <a href="https://github.com/aryopg/mmlu-redux">官方仓库</a> · <a href="https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0">数据集</a>。规模按 MMLU-Redux 2.0 统计，包含 57 个学科、每科 100 条测试样本；6.49% 是对原 MMLU 题目错误比例的估计。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mmlu-pro-citation" title="展开或收起引用">MMLU-Pro</button>
        </td>
        <td>2024 年</td>
        <td>MMLU 的增强版，覆盖 <b>14 个学科领域</b>，通过引入更复杂的问题、将选项从 <b>4 个增加至 10 个</b>，并过滤简单及存在噪声的题目，提高评测的<b>区分度</b>。</td>
        <td>12,032</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="mmlu-pro-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="mmlu-pro-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/mmlu-pro-qwen.liquid %}
        </td>
      </tr>
      <tr id="mmlu-pro-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Wang et al. <em>MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.</em> arXiv:2406.01574, 2024. <a href="https://arxiv.org/abs/2406.01574">论文链接</a> · <a href="https://github.com/TIGER-AI-Lab/MMLU-Pro">官方仓库</a> · <a href="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/viewer/default/test">测试集</a>。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="agieval-citation" title="展开或收起引用">AGIEval</button>
        </td>
        <td>2023 年</td>
        <td>以<b>人类标准化考试</b>为基础构建的综合评测基准，涵盖中国高考、SAT、LSAT、数学竞赛及公务员考试等，覆盖<b>语言理解、数学推理、逻辑推理和多学科知识</b>（客观选择题和填空题）。</td>
        <td title="原论文统计的评测样本规模。">8,062</td>
        <td title="按原论文：选择题使用Accuracy，填空题使用EM和F1。">Accuracy<br>EM / F1</td>
      </tr>
      <tr id="agieval-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Zhong et al. <em>AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models.</em> arXiv:2304.06364, 2023. <a href="https://arxiv.org/abs/2304.06364">论文链接</a> · <a href="https://github.com/ruixiangcui/AGIEval">官方仓库</a> · <a href="https://github.com/ruixiangcui/AGIEval/tree/main/data/v1">原版数据</a>。规模按原论文 §3.3.1 的 8,062 道评测题统计；仓库后续版本及输入变体的文件条数可能不同。指标按原论文区分：选择题使用 Accuracy，填空题使用 EM / F1。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ceval-citation" title="展开或收起引用">C-Eval</button>
        </td>
        <td>2023 年</td>
        <td>C-Eval（Chinese Evaluation Suite）是面向<b>中文语境</b>的综合性学科四选一评测基准，覆盖 STEM、人文、社会科学等 <b>52 个学科</b>，包含<b>初中、高中、大学和专业</b>四个难度层级。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="ceval-examples" title="展开或收起文字样例">[样例]</button></td>
        <td title="52个学科的test划分合计；不含dev和val。">12,342</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="ceval-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="ceval-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/ceval-examples.liquid %}
        </td>
      </tr>
      <tr id="ceval-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/ceval-qwen.liquid %}
        </td>
      </tr>
      <tr id="ceval-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Huang et al. <em>C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models.</em> NeurIPS, 2023. <a href="https://arxiv.org/abs/2305.08322">论文链接</a> · <a href="https://cevalbenchmark.com/">官方网站</a> · <a href="https://huggingface.co/datasets/ceval/ceval-exam">数据集</a>。规模为 52 个学科的 test 划分合计 12,342 题，不含 dev 和 val；13,948 为所有划分的总题数。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="cmmlu-citation" title="展开或收起引用">CMMLU</button>
        </td>
        <td>2023 年</td>
        <td>面向中文语境的综合性多学科知识评测基准，覆盖 STEM、人文、社会科学及其他领域的 <b>67 个学科</b>。相比 C-Eval，CMMLU 更注重<b>中国特有的文化与区域知识</b>，涵盖中国饮食文化、驾驶规则、古汉语等特色主题，并兼顾全球通用知识，用于评估模型在中文语境下的<b>知识储备与推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="cmmlu-examples" title="展开或收起文字样例">[样例]</button></td>
        <td title="官方仓库data/test的67个学科CSV合计，不含dev。">11,582</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="cmmlu-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="cmmlu-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/cmmlu-examples.liquid %}
        </td>
      </tr>
      <tr id="cmmlu-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/cmmlu-qwen.liquid %}
        </td>
      </tr>
      <tr id="cmmlu-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Li et al. <em>CMMLU: Measuring Massive Multitask Language Understanding in Chinese.</em> arXiv:2306.09212, 2023. <a href="https://arxiv.org/abs/2306.09212">论文链接</a> · <a href="https://github.com/haonan-li/CMMLU">官方仓库</a> · <a href="https://github.com/haonan-li/CMMLU/tree/master/data/test">测试集</a>。规模按官方仓库 data/test 中 67 个学科的 CSV 样本统计，合计 11,582 题，不含用于少样本提示的 dev 划分。</td>
      </tr>
    </tbody>
  </table>
</div>

<br>

## 3. 事实真实性与可靠性评测

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="事实真实性与可靠性评测基准">
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
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="truthfulqa-citation" title="展开或收起引用">TruthfulQA</button>
        </td>
        <td>2021 年</td>
        <td>专门针对<b>人类常见误解</b>设计的问答基准，覆盖健康、法律、金融、政治等 <b>38 个类别</b>，通过容易引发常见误解的问题，评估模型回答的<b>真实性与准确性</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="truthfulqa-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>817</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="truthfulqa-performance" title="展开或收起 Qwen 系列模型的得分柱状图">Truthfulness</button></td>
      </tr>
      <tr id="truthfulqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/truthfulqa-examples.liquid %}
        </td>
      </tr>
      <tr id="truthfulqa-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/truthfulqa-qwen.liquid %}
        </td>
      </tr>
      <tr id="truthfulqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Lin, Hilton, and Evans. <em>TruthfulQA: Measuring How Models Mimic Human Falsehoods.</em> ACL, 2022（预印本于 2021 年发布）。<a href="https://arxiv.org/abs/2109.07958">论文链接</a> · <a href="https://github.com/sylinrl/TruthfulQA">官方仓库</a> · <a href="https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv">评测题目</a>。</td>
      </tr>
    </tbody>
  </table>
</div>

<br>

## 4. 数学推理

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="数学推理基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="gsm8k-citation" title="展开或收起引用">GSM8K</button></td>
        <td>2021 年</td>
        <td>GSM8K（Grade School Math 8K）是<b>小学数学应用题基准</b>，主要评估模型的<b>多步算术推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="gsm8k-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>1,319</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="gsm8k-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="gsm8k-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/gsm8k-examples.liquid %}
        </td>
      </tr>
      <tr id="gsm8k-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/gsm8k-qwen.liquid %}
        </td>
      </tr>
      <tr id="gsm8k-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Cobbe et al. <em>Training Verifiers to Solve Math Word Problems.</em> 2021。<a href="https://arxiv.org/abs/2110.14168">论文链接</a> · <a href="https://github.com/openai/grade-school-math">官方仓库</a> · <a href="https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl">评测题目</a>。规模采用官方测试划分的 1,319 道题，不含 7,473 道训练题；准确率按最终答案评估。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="math-citation" title="展开或收起引用">MATH</button></td>
        <td>2021 年</td>
        <td>MATH（Mathematics Aptitude Test of Heuristics）是<b>高中竞赛数学评测基准</b>，覆盖代数、几何、数论、组合与概率等 <b>7 个数学领域</b>，包含 <b>5 个难度等级</b>，主要评估模型的<b>复杂数学推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="math-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>5,000</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="math-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="math-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/math-examples.liquid %}
        </td>
      </tr>
      <tr id="math-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/math-qwen.liquid %}
        </td>
      </tr>
      <tr id="math-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Hendrycks et al. <em>Measuring Mathematical Problem Solving With the MATH Dataset.</em> NeurIPS, 2021。<a href="https://arxiv.org/abs/2103.03874">论文链接</a> · <a href="https://github.com/hendrycks/math">官方仓库</a> · <a href="https://huggingface.co/datasets/qwedsacf/competition_math">数据集</a>。规模采用原始 MATH 的 5,000 道测试题，不含 7,500 道训练题；准确率按最终答案评估。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="aime-citation" title="展开或收起引用">AIME</button></td>
        <td>2024 年<br>2025 年</td>
        <td>AIME（American Invitational Mathematics Examination，美国数学邀请赛）是<b>高中竞赛数学评测</b>，题目涉及代数、几何、数论与组合等领域，答案为 <b>0–999 的整数</b>，主要评估模型的<b>高难度数学推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="aime-examples" title="展开或收起 AIME 真题样例">[样例]</button></td>
        <td>30 / 年</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="aime-performance" title="展开或收起 Qwen3-8B 在 AIME 2024 与 AIME 2025 的得分柱状图">Accuracy</button></td>
      </tr>
      <tr id="aime-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/aime-examples.liquid %}
        </td>
      </tr>
      <tr id="aime-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/aime-qwen.liquid %}
        </td>
      </tr>
      <tr id="aime-citation" class="benchmark-citation-row" hidden>
        <td colspan="5"><em>American Invitational Mathematics Examination (AIME).</em> <a href="https://maa.org/student-programs/amc/american-invitational-mathematics-examination-aime/">MAA 官方介绍</a> · <a href="https://huggingface.co/datasets/HuggingFaceH4/aime_2024">AIME 2024 公开题集</a> · <a href="https://huggingface.co/datasets/math-ai/aime25">AIME 2025 公开题集</a>。年份指本条采用的试题版本，而非赛事创办时间；每年合并 AIME I 与 AIME II 各 15 道题，共 30 道题。准确率按最终整数答案评估；图表分别保留两届成绩，不合并平均，也不额外指定采样次数或推理设置。</td>
      </tr>
    </tbody>
  </table>
</div>

<br>

## 5. 代码生成

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="代码生成基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="humaneval-citation" title="展开或收起引用">HumanEval</button></td>
        <td>2021 年</td>
        <td>人工编写的 <b>Python 函数生成基准</b>，要求模型根据自然语言描述实现函数，并通过<b>单元测试</b>验证代码的<b>功能正确性</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="humaneval-examples" title="展开或收起代码样例">[样例]</button></td>
        <td>164</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="humaneval-performance" title="展开或收起 Qwen 系列模型的 Pass@1 柱状图">Pass@1</button></td>
      </tr>
      <tr id="humaneval-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/humaneval-examples.liquid %}
        </td>
      </tr>
      <tr id="humaneval-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/humaneval-qwen.liquid %}
        </td>
      </tr>
      <tr id="humaneval-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Chen et al. <em>Evaluating Large Language Models Trained on Code.</em> 2021。<a href="https://arxiv.org/abs/2107.03374">论文链接</a> · <a href="https://github.com/openai/human-eval">官方仓库</a> · <a href="https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz">评测题目</a>。Pass@1 表示单次生成的代码通过该题全部单元测试的比例；样例为 HumanEval/121，包含题目说明与参考实现。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mbpp-citation" title="展开或收起引用">MBPP</button></td>
        <td>2021 年</td>
        <td>MBPP（Mostly Basic Programming Problems）是面向<b>入门程序员</b>的 Python 代码生成基准，要求模型根据自然语言描述生成函数，并通过<b>单元测试</b>验证正确性。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mbpp-examples" title="展开或收起代码样例">[样例]</button></td>
        <td>500</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="mbpp-performance" title="展开或收起 Qwen 系列模型的 Pass@1 柱状图">Pass@1</button></td>
      </tr>
      <tr id="mbpp-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/mbpp-examples.liquid %}
        </td>
      </tr>
      <tr id="mbpp-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/mbpp-qwen.liquid %}
        </td>
      </tr>
      <tr id="mbpp-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Austin et al. <em>Program Synthesis with Large Language Models.</em> 2021。<a href="https://arxiv.org/abs/2108.07732">论文链接</a> · <a href="https://github.com/google-research/google-research/tree/master/mbpp">官方仓库</a> · <a href="https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl">评测题目</a>。规模采用原始 MBPP 的 500 道测试题（task_id 11–510），不混入训练、验证及提示示例，也不采用 sanitized 子集规模；样例对应 task_id 56，示例实现按截图展示。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="livecodebench-citation" title="展开或收起引用">LiveCode<wbr>Bench</button></td>
        <td>2024 年</td>
        <td>针对 HumanEval 的<b>数据污染、难度饱和</b>等问题，从 LeetCode、AtCoder 和 Codeforces 持续收集新竞赛题目，通过<b>时间划分</b>降低数据污染风险，并扩展至<b>代码修复、执行预测及测试输出预测</b>等任务。</td>
        <td>511</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="livecodebench-performance" title="展开或收起 Qwen 系列模型的 Pass@1 柱状图">Pass@1</button></td>
      </tr>
      <tr id="livecodebench-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/livecodebench-qwen.liquid %}
        </td>
      </tr>
      <tr id="livecodebench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Jain et al. <em>LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.</em> 2024。<a href="https://arxiv.org/abs/2403.07974">论文链接</a> · <a href="https://livecodebench.github.io/">项目主页</a> · <a href="https://github.com/LiveCodeBench/LiveCodeBench#dataset-versions">官方仓库与版本说明</a>。规模采用代码生成任务的 release_v2：511 道题，题目发布时间覆盖 2023 年 5 月至 2024 年 5 月；该基准持续更新，其他版本规模不同。此处规模说明不代表图中模型成绩采用同一版本或时间窗口。</td>
      </tr>
    </tbody>
  </table>
</div>

<br>

## 6. 专家级科学推理

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="专家级科学推理基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="gpqa-citation" title="展开或收起引用">GPQA</button></td>
        <td>2023 年</td>
        <td>GPQA（Graduate-Level Google-Proof Q&amp;A Benchmark）是由<b>领域专家</b>编写的<b>研究生级科学问答四选一基准</b>，涵盖生物、物理和化学，重点评估<b>专业科学知识与推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="gpqa-examples" title="展开或收起文字样例">[样例]</button></td>
        <td>448</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="gpqa-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="gpqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/gpqa-examples.liquid %}
        </td>
      </tr>
      <tr id="gpqa-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/gpqa-qwen.liquid %}
        </td>
      </tr>
      <tr id="gpqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Rein et al. <em>GPQA: A Graduate-Level Google-Proof Q&amp;A Benchmark.</em> COLM, 2024（预印本于 2023 年发布）。<a href="https://arxiv.org/abs/2311.12022">论文链接</a> · <a href="https://github.com/idavidrein/gpqa">官方仓库</a> · <a href="https://huggingface.co/datasets/idavidrein/gpqa">数据集</a>。规模采用 GPQA Main 的 448 道题，不与 198 道题的 Diamond 子集混用；图中成绩保留给定模型名称与分数，不据此推断具体子集或提示设置。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="gpqa-diamond-citation" title="展开或收起引用">GPQA Diamond</button></td>
        <td>2023 年</td>
        <td>GPQA 的<b>高质量困难子集</b>，要求领域专家认可答案且多数非专家无法正确作答，重点评估<b>高难度科学推理能力</b>。</td>
        <td>198</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="gpqa-diamond-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="gpqa-diamond-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/gpqa-diamond-qwen.liquid %}
        </td>
      </tr>
      <tr id="gpqa-diamond-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Rein et al. <em>GPQA: A Graduate-Level Google-Proof Q&amp;A Benchmark.</em> COLM, 2024（预印本于 2023 年发布）。<a href="https://arxiv.org/abs/2311.12022">论文链接</a> · <a href="https://github.com/idavidrein/gpqa">官方仓库</a> · <a href="https://huggingface.co/datasets/idavidrein/gpqa/viewer/gpqa_diamond">Diamond 数据集</a>。规模采用 GPQA Diamond 的 198 道题，与 GPQA Main 的 448 道题分别列示。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="supergpqa-citation" title="展开或收起引用">SuperGPQA</button></td>
        <td>2025 年</td>
        <td>SuperGPQA 是覆盖 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="supergpqa-disciplines" title="展开或收起13个学科门类">13 个大学科门类</button>、<b>285 个研究生级细分学科</b>的大规模多学科问答基准，包含约 <b>2.65 万道选择题</b>，平均每题 <b>9.67 个选项</b>。数据经<b>人类专家与大模型协同筛选</b>，重点评估模型在长尾专业领域的<b>知识理解与复杂推理能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="supergpqa-examples" title="展开或收起研究生级问答样例">[样例]</button></td>
        <td>26,529</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="supergpqa-performance" title="展开或收起 Qwen 系列模型的 Accuracy 柱状图">Accuracy</button></td>
      </tr>
      <tr id="supergpqa-disciplines" class="benchmark-datasets-row" hidden>
        <td colspan="5">
          {% include benchmarks/supergpqa-disciplines.liquid %}
        </td>
      </tr>
      <tr id="supergpqa-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/supergpqa-examples.liquid %}
        </td>
      </tr>
      <tr id="supergpqa-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/supergpqa-qwen.liquid %}
        </td>
      </tr>
      <tr id="supergpqa-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Du et al. <em>SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines.</em> 2025。<a href="https://arxiv.org/abs/2502.14739">论文</a> · <a href="https://github.com/SuperGPQA/SuperGPQA">官方仓库</a> · <a href="https://huggingface.co/datasets/m-a-p/SuperGPQA">数据集</a> </td>
      </tr>
    </tbody>
  </table>
</div>

<br>

## 7. 对话与人类偏好评测

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="对话与人类偏好评测基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="mt-bench-citation" title="展开或收起引用">MT-Bench</button></td>
        <td>2023 年</td>
        <td>面向<b>开放式对话</b>的大语言模型评测基准，包含 <b>80 组两轮问题</b>，覆盖写作、角色扮演、信息抽取、推理、数学、代码、STEM 与人文社科等 <b>8 类任务</b>，重点评估模型的<b>多轮对话与指令遵循能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="mt-bench-examples" title="展开或收起两轮对话样例">[样例]</button></td>
        <td>80</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="mt-bench-performance" title="展开或收起 Qwen 系列模型的 Judge Score 柱状图">Judge Score</button></td>
      </tr>
      <tr id="mt-bench-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/mt-bench-examples.liquid %}
        </td>
      </tr>
      <tr id="mt-bench-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/mt-bench-qwen.liquid %}
        </td>
      </tr>
      <tr id="mt-bench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Zheng et al. <em>Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.</em> NeurIPS, 2023。<a href="https://arxiv.org/abs/2306.05685">论文链接</a> · <a href="https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge">官方仓库</a> · <a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl">评测题目</a>。规模按 80 组两轮问题计，共 160 个轮次；Judge Score 为模型裁判评分，满分 10 分，不是百分比。样例题目对应 question_id 155，回答与省略部分按截图展示。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="chatbot-arena-citation" title="展开或收起引用">Chatbot Arena</button></td>
        <td>2023 年</td>
        <td>Chatbot Arena 是基于<b>匿名模型两两对战</b>的众包人类偏好评测平台，用户同时与两个匿名模型交互并选择更优回答，用于评估模型在<b>真实开放式对话场景中的人类偏好</b>。</td>
        <td>30K</td>
        <td>Win Rate<br>Elo</td>
      </tr>
      <tr id="chatbot-arena-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Zheng et al. <em>Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.</em> NeurIPS, 2023。<a href="https://arxiv.org/abs/2306.05685">论文链接</a> · <a href="https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge">官方仓库</a> · <a href="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations">早期公开对话数据</a>。30K 指该论文公开的约 3 万条带有人类偏好标注的对话，并非固定测试题数或当前平台累计规模。Win Rate 表示两两对战胜率；Elo 是基于对战结果估计的相对评分，不是百分比。</td>
      </tr>
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="ifeval-citation" title="展开或收起引用">IFEval</button></td>
        <td>2023 年</td>
        <td>IFEval（Instruction-Following Eval）是面向大语言模型<b>指令遵循能力</b>的评测基准，包含 <b>541 个带有可验证指令的 Prompt</b>，覆盖 <b>25 类约束</b>，包括关键词、语言、长度、格式及标点等，重点评估模型对<b>明确、可验证指令的执行能力</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="ifeval-examples" title="展开或收起指令样例">[样例]</button></td>
        <td>541</td>
        <td><button type="button" class="benchmark-performance-toggle" aria-expanded="false" aria-controls="ifeval-performance" title="展开或收起 Qwen 系列模型的 IFEval 得分柱状图">Accuracy</button></td>
      </tr>
      <tr id="ifeval-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/ifeval-examples.liquid %}
        </td>
      </tr>
      <tr id="ifeval-performance" class="benchmark-performance-row" hidden>
        <td colspan="5">
          {% include charts/ifeval-qwen.liquid %}
        </td>
      </tr>
      <tr id="ifeval-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Zhou et al. <em>Instruction-Following Evaluation for Large Language Models.</em> 2023。<a href="https://arxiv.org/abs/2311.07911">论文链接</a> · <a href="https://github.com/google-research/google-research/tree/master/instruction_following_eval">官方仓库</a> · <a href="https://github.com/google-research/google-research/blob/master/instruction_following_eval/data/input_data.jsonl">评测题目</a>。规模按官方公开的 541 个 Prompt 计，覆盖 25 类可验证指令。评测区分 Prompt 级与 Instruction 级准确率，并提供 strict 与 loose 两种判定方式；此处图表不额外指定具体口径。样例对应官方数据 key 2637。</td>
      </tr>
    </tbody>
  </table>
</div>

<br>
## 8. 安全性与对齐评测

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="安全性评测基准">
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
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="do-not-answer-citation" title="展开或收起引用">Do-Not-Answer</button></td>
        <td>2023 年</td>
        <td>面向大语言模型<b>安全防护能力</b>的评测基准，仅包含<b>不应被模型直接遵从的有害或风险性问题</b>，覆盖隐私与敏感信息泄露、恶意用途、歧视与仇恨、错误信息及人机交互风险等 <b>5 大领域</b>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="do-not-answer-examples" title="展开或收起安全评测问题样例">[样例]</button></td>
        <td>939</td>
        <td>Accuracy</td>
      </tr>
      <tr id="do-not-answer-examples" class="benchmark-examples-row" hidden>
        <td colspan="5">
          {% include benchmarks/do-not-answer-examples.liquid %}
        </td>
      </tr>
      <tr id="do-not-answer-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">Wang et al. <em>Do-Not-Answer: Evaluating Safeguards in LLMs.</em> Findings of EACL, 2024（预印本发布于 2023 年）。<a href="https://arxiv.org/abs/2308.13387">预印本</a> · <a href="https://aclanthology.org/2024.findings-eacl.61/">正式论文</a> · <a href="https://github.com/Libr-AI/do-not-answer">官方仓库</a>。规模按英文原版的 939 条指令计，覆盖 5 个风险领域、12 类危害；不包含后续中文扩展集。此处 Accuracy 指安全回答比例，即未产生有害回答的占比，而非对这些风险请求给出实质性答案的比例。</td>
      </tr>
    </tbody>
  </table>
</div>

<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
<script>
  window.buildNaturalQuestionsQwenOption = (
    {% include charts/natural-questions-qwen.chart.js %}
  );
  window.buildWinoGrandeQwenOption = (
    {% include charts/winogrande-qwen.chart.js %}
  );
  window.buildHellaSwagQwenOption = (
    {% include charts/hellaswag-qwen.chart.js %}
  );
  window.buildBBHQwenOption = (
    {% include charts/bbh-qwen.chart.js %}
  );
  window.buildARCQwenOption = (
    {% include charts/arc-qwen.chart.js %}
  );
  window.buildTheoremQAQwenOption = (
    {% include charts/theoremqa-qwen.chart.js %}
  );
  window.buildMMLUQwenOption = (
    {% include charts/mmlu-qwen.chart.js %}
  );
  window.buildMMLUProQwenOption = (
    {% include charts/mmlu-pro-qwen.chart.js %}
  );
  window.buildCEvalQwenOption = (
    {% include charts/ceval-qwen.chart.js %}
  );
  window.buildCMMLUQwenOption = (
    {% include charts/cmmlu-qwen.chart.js %}
  );
  window.buildTruthfulQAQwenOption = (
    {% include charts/truthfulqa-qwen.chart.js %}
  );
  window.buildGSM8KQwenOption = (
    {% include charts/gsm8k-qwen.chart.js %}
  );
  window.buildMATHQwenOption = (
    {% include charts/math-qwen.chart.js %}
  );
  window.buildAIMEQwenOption = (
    {% include charts/aime-qwen.chart.js %}
  );
  window.buildHumanEvalQwenOption = (
    {% include charts/humaneval-qwen.chart.js %}
  );
  window.buildMBPPQwenOption = (
    {% include charts/mbpp-qwen.chart.js %}
  );
  window.buildLiveCodeBenchQwenOption = (
    {% include charts/livecodebench-qwen.chart.js %}
  );
  window.buildGPQAQwenOption = (
    {% include charts/gpqa-qwen.chart.js %}
  );
  window.buildGPQADiamondQwenOption = (
    {% include charts/gpqa-diamond-qwen.chart.js %}
  );
  window.buildSuperGPQAQwenOption = (
    {% include charts/supergpqa-qwen.chart.js %}
  );
  window.buildMTBenchQwenOption = (
    {% include charts/mt-bench-qwen.chart.js %}
  );
  window.buildIFEvalQwenOption = (
    {% include charts/ifeval-qwen.chart.js %}
  );
</script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
