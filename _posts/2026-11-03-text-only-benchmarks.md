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
  .benchmark-citation-toggle, .benchmark-datasets-toggle, .benchmark-performance-toggle, .benchmark-examples-toggle { padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; font-weight: bold; text-decoration: none; cursor: pointer; }
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
        <td>The Pile 的独立测试集，覆盖 <strong><button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="pile-datasets" title="展开或收起22个子数据集名称">22</button> 个多领域子数据集</strong>，涵盖学术论文、书籍、代码、网页、法律文档及在线对话等，主要用于评估语言模型的<strong>跨领域语言建模能力</strong>。</td>
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
        <td>问题来自<strong>真实 Google 搜索查询</strong>，要求模型结合 Wikipedia 文档定位长答案并提取短答案，同时判断文档中是否存在答案，评估模型的<strong>长文档理解、答案定位与问答能力</strong>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="natural-questions-examples" title="展开或收起三个文字样例">[样例]</button></td>
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
        <td>问题由<strong>知识问答爱好者编写</strong>，配有从 Wikipedia 和网页收集的证据文档。问题与证据之间常存在词汇差异，需要整合多句话定位答案，评估模型的<strong>事实知识、阅读理解与跨句推理能力</strong>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="triviaqa-examples" title="展开或收起文字样例">[样例]</button></td>
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
        <td>围绕<strong>日常生活中的物理常识与操作任务</strong>，给定一个目标和两个候选方案，要求选出更合理、可行的方案。评估模型对<strong>物体属性、工具用途及物理交互</strong>的理解与常识推理能力。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="piqa-examples" title="展开或收起文字样例">[样例]</button></td>
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
        <td>以<strong>指代消解</strong>（判断代词
          如 it、she 指代句中的哪个人或物）为核心，给定句子与两个候选对象，要求根据语境补全空缺、确定正确指代。采用<strong>对抗过滤</strong>减少词语关联等数据偏差，评估模型的<strong>常识推理与语境理解能力</strong>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="winogrande-examples" title="展开或收起两个文字样例">[样例]</button></td>
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
        <td>围绕<strong>日常活动与事件续写</strong>，给定一段情境描述，从四个候选续写中选择最合理的后续情节。干扰选项通过<strong>对抗过滤</strong>筛选，语法自然但语义不合常理，评估模型的<strong>事件预测、情境理解与常识推理能力</strong>。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="hellaswag-examples" title="展开或收起文字样例">[样例]</button></td>
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
        <td>由研究社区协作构建的<strong>综合能力评测合集</strong>，论文收录 <strong>204 个任务</strong>，覆盖语言理解、数学、常识推理、代码、多语言与社会偏差等领域。包含静态题目与可编程交互任务，用于考察模型的<strong>多维能力、泛化表现与规模增长规律</strong>。</td>
        <td title="不同子任务提供不同数量的评测样本，部分任务支持程序化生成；没有统一测试集规模。">依任务而定</td>
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
        <td>BIG-Bench 的<strong>轻量代表性子集</strong>，精选 <strong><button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="big-bench-lite-tasks" title="展开或收起24个任务名称">24</button> 个 JSON 格式任务</strong>，覆盖推理、代码、多语言及偏差测量等能力。不包含需要复杂交互的可编程任务，便于以<strong>更低评测成本</strong>比较模型的综合表现；与 BIG-Bench Hard（BBH）不同。</td>
        <td title="24 是任务数量，不是测试样本数量；各子任务的样本数和评测采样设置不同。">依任务而定</td>
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
</script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
