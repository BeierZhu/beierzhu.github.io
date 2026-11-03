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
  .benchmark-performance-panel { position: sticky; left: 0; width: min(100%, calc(100vw - 60px)); }
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
        <td>围绕<strong>日常生活中的物理常识与操作任务</strong>，给定一个目标（Goal）和两个候选方案，要求选出更合理、可行的方案。评估模型对<strong>物体属性、工具用途及物理交互</strong>的理解与常识推理能力。<button type="button" class="benchmark-examples-toggle" aria-expanded="false" aria-controls="piqa-examples" title="展开或收起文字样例">[样例]</button></td>
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
    </tbody>
  </table>
</div>


<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
<script>
  window.buildNaturalQuestionsQwenOption = (
    {% include charts/natural-questions-qwen.chart.js %}
  );
</script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
