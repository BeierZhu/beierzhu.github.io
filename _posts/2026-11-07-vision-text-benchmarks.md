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
  .benchmark-subtask-table tbody:last-child tr:last-child th, .benchmark-subtask-table tbody:last-child tr:last-child td { border-bottom: 0; }
---

## 1. 综合视觉理解


<div class="table-responsive benchmark-preview-table" markdown="0">
  <table aria-label="视觉文本基准">
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
        <td>包含<b>感知（Perception）和认知（Cognition）</b>两大维度、共 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mme-subtasks" title="展开或收起14项子任务">14 项子任务</button>。所有题目均采用 <b>Yes/No 二分类</b>形式，每张图像对应一正一负两道问题，最终以汇总分数 <b>Score</b> 衡量模型表现。</td>
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
      <tr class="benchmark-data-row">
        <td><button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="seed-bench-citation" title="展开或收起引用">SEED-Bench</button></td>
        <td>2023 年</td>
        <td>面向多模态大语言模型的<b>图像与视频理解</b>评测基准，涵盖场景理解、物体识别、空间关系、视觉推理、动作识别、动作预测及过程理解等 <b>12 个评测维度</b>。采用人工验证的<b>四选一选择题</b>，评估模型的视觉感知、空间推理与时序理解能力。</td>
        <td class="benchmark-count-cell">19,242</td>
        <td class="benchmark-metric-cell">Accuracy</td>
      </tr>
      <tr id="seed-bench-citation" class="benchmark-citation-row" hidden>
        <td colspan="5">
          Li et al. <em>SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension.</em> arXiv:2307.16125, 2023.
          <a href="https://arxiv.org/abs/2307.16125">论文链接</a> ·
          <a href="https://github.com/AILab-CVC/SEED-Bench">官方仓库</a>。
          <p>注：规模采用原论文 Table 1 的 19,242 道选择题，包含 9 个图像空间理解维度与 3 个视频时序理解维度。原论文采用候选答案似然排序确定预测选项，并以 Accuracy 评估。</p>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
