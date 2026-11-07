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

本文按照《大模型测评基准I：纯文本基准》的结构，逐项整理视觉文本评测基准的简介、规模、评测指标、样例与模型表现。

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
        <td>MME（Multimodal Large Language Model Evaluation）是面向多模态大语言模型的综合图像理解评测基准，包含感知（Perception）和认知（Cognition）两大维度、共 <button type="button" class="benchmark-datasets-toggle" aria-expanded="false" aria-controls="mme-subtasks" title="展开或收起14项子任务">14 项子任务</button>，涵盖物体识别、计数、位置、颜色、OCR、电影海报、名人、场景、地标与艺术作品识别，以及常识推理、数值计算、文本翻译和代码推理。所有题目均采用 Yes/No 二分类形式，每张图像对应一正一负两道问题，通过 Accuracy 和 Accuracy+ 评估模型的视觉感知与推理能力。</td>
        <td class="benchmark-count-cell">1,187 张图像<br>2,374 道 Yes/No 题</td>
        <td class="benchmark-metric-cell">Accuracy / Accuracy+</td>
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
        <td colspan="5">Fu et al. <em>MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models.</em> arXiv:2306.13394, 2023。<a href="https://arxiv.org/abs/2306.13394">论文链接</a>。</td>
      </tr>
    </tbody>
  </table>
</div>

<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/benchmark-performance.js' | relative_url }}" defer></script>
