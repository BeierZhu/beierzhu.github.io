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
  .benchmark-preview-table th { white-space: nowrap; text-align: left; vertical-align: middle; padding: .5rem 0; border-top: 1px solid var(--global-divider-color); border-bottom: 1px solid var(--global-divider-color); }
  .benchmark-preview-table td { vertical-align: top; }
  .benchmark-data-row td:not(:nth-child(3)) { white-space: nowrap; }
  .benchmark-citation-toggle { padding: 0; border: 0; background: none; color: var(--global-theme-color); font: inherit; font-weight: bold; text-decoration: none; cursor: pointer; }
  .benchmark-citation-row[hidden] { display: none; }
  .benchmark-citation-row td { padding: .75rem; box-shadow: inset 0 1px 0 var(--global-divider-color); white-space: normal; overflow-wrap: anywhere; font-size: .85rem; line-height: 1.6; }
---


## 1. 通用推理

<div class="table-responsive benchmark-preview-table" markdown="0">
  <table>
    <colgroup>
      <col style="width: 7%">
      <col style="width: 10%">
      <col style="width: 57%">
      <col style="width: 8%">
      <col style="width: 12%">
      <col style="width: 6%">
    </colgroup>
    <thead>
      <tr>
        <th scope="col">名称</th>
        <th scope="col">发布时间</th>
        <th scope="col">简介</th>
        <th scope="col">规模</th>
        <th scope="col">评测指标</th>
        <th scope="col">SoTA</th>
      </tr>
    </thead>
    <tbody>
      <tr class="benchmark-data-row">
        <td>
          <button type="button" class="benchmark-citation-toggle" aria-expanded="false" aria-controls="pile-citation" title="展开或收起引用">Pile</button>
        </td>
        <td>2020 年</td>
        <td>由 <strong>22 个多领域子数据集</strong>组成的大规模英文文本语料库，涵盖学术论文、书籍、代码、网页、法律文档及在线对话等，主要用于评估语言模型的<strong>跨领域语言建模能力</strong>。</td>
        <td>825 GB</td>
        <td>PPL / BPB</td>
        <td>—</td>
      </tr>
      <tr id="pile-citation" class="benchmark-citation-row" hidden>
        <td colspan="6">Gao et al. <em>The Pile: An 800GB Dataset of Diverse Text for Language Modeling.</em> arXiv:2101.00027. <a href="https://arxiv.org/abs/2101.00027">论文链接</a>。</td>
      </tr>
    </tbody>
  </table>
</div>


<script src="{{ '/assets/js/benchmark-citations.js' | relative_url }}" defer></script>
