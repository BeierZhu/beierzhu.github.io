---
layout: page
permalink: /publications/
title: publications
titledisplay: Publications
description:
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<p class="post-description language-content language-block" data-language-content="en">Published {{ site.data.summary_stats.total }} papers in top venues, including {{ site.data.summary_stats.first_corresponding }} as first or corresponding author and {{ site.data.summary_stats.distinguished }} Oral/Spotlight/Highlight.</p>
<p class="post-description language-content language-block" data-language-content="zh">已发表 {{ site.data.summary_stats.total }} 篇论文，其中 {{ site.data.summary_stats.first_corresponding }} 篇为第一作者或通讯作者，{{ site.data.summary_stats.distinguished }} 篇获得 Oral/Spotlight/Highlight。</p>

<!-- Bibsearch Feature -->

 <i class="fa-solid fa-handshake" style="font-size: 0.7em; vertical-align: super; margin-left: 1px;"></i> and <i class="fa-solid fa-envelope" style="font-size: 0.7em; vertical-align: super; margin-left: 1px;"></i> denote equal contribution and corresponding authorship. You can find full list of my publications on my [Google Scholar](https://scholar.google.com/citations?hl=en&user=jHczmjwAAAAJ). 

<div class="stats-tables-container">
<div class="venue-stats-table">
<table>
  <caption data-i18n="publications.byVenue">By Venue and Authorship</caption>
  <tr class="total-row">
      <td><b data-i18n="publications.venue">Venue</b></td>
      <td><b data-i18n="publications.papers">Papers</b></td>
      <td><b>
        <span class="language-content language-inline" data-language-content="en">1<sup>st</sup> and</span>
        <span class="language-content language-inline" data-language-content="zh">第一作者及</span>
        <i class="fa-solid fa-envelope"></i>
      </b></td>
    </tr>
  <tbody>
    {% assign total_count = 0 %}
    {% assign total_fc = 0 %}
    {% for venue in site.data.venue_stats %}
    {% assign total_count = total_count | plus: venue.count %}
    {% assign total_fc = total_fc | plus: venue.first_corresponding %}
    <tr>
      <td>{{ venue.venue }}</td>
      <td>{{ venue.count }}</td>
      <td>{{ venue.first_corresponding }}</td>
    </tr>
    {% endfor %}
    <tr class="total-row">
       <td><b data-i18n="publications.total">Total</b></td>
      <td><b>{{ total_count }}</b></td>
      <td><b>{{ total_fc }}</b></td>
    </tr>
  </tbody>
</table>
</div>

<div class="class-stats-table">
<table>
  <caption data-i18n="publications.byTopic">By Research Topic</caption>
  <tr class="total-row">
    <td><b data-i18n="publications.category">Category</b></td>
    <td><b data-i18n="publications.topic">Topic</b></td>
    <td><b data-i18n="publications.papers">Papers</b></td>
  </tr>
  <tbody>
    {% for supclass in site.data.class_stats %}
    {% if supclass.supclass == "Others" %}
    <tr>
      <td><b>{{ supclass.supclass }}</b></td>
      <td></td>
      <td>{{ supclass.total }}</td>
    </tr>
    {% else %}
    {% assign first_row = true %}
    {% for cls in supclass.classes %}
    <tr>
      {% if first_row %}
      <td rowspan="{{ supclass.classes.size }}"><b>{{ supclass.supclass }}</b><br>({{ supclass.total }})</td>
      {% assign first_row = false %}
      {% endif %}
      <td>{{ cls.name }}</td>
      <td>{{ cls.count }}</td>
    </tr>
    {% endfor %}
    {% endif %}
    {% endfor %}
  </tbody>
</table>
</div>
</div>

{% include bib_search.liquid %}

<div class="publications">

{% bibliography %}

</div>
