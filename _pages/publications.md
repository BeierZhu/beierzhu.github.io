---
layout: page
permalink: /publications/
title: publications
titledisplay: Publications
description: Published 30+ papers in top venues on robust learning and multimodal learning, including 20+ as first or corresponding author.
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->


You can find full list of my publications and recent works on my [Google Scholar](https://scholar.google.com/citations?hl=en&user=jHczmjwAAAAJ). 
 <i class="fa-solid fa-handshake" style="font-size: 0.7em; vertical-align: super; margin-left: 1px;"></i> and <i class="fa-solid fa-envelope" style="font-size: 0.7em; vertical-align: super; margin-left: 1px;"></i> denote equal contribution and corresponding authorship.

<div class="stats-tables-container">
<div class="venue-stats-table">
<table>
  <caption>By Venue</caption>
  <tr class="total-row">
      <td><b>Venue</b></td>
      <td><b>Papers</b></td>
      <td><b>1<sup>st</sup>/<i class="fa-solid fa-envelope"></i></b></td>
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
      <td><b>Total</b></td>
      <td><b>{{ total_count }}</b></td>
      <td><b>{{ total_fc }}</b></td>
    </tr>
  </tbody>
</table>
</div>

<div class="class-stats-table">
<table>
  <caption>By Research Topic</caption>
  <tr class="total-row">
    <td><b>Category</b></td>
    <td><b>Topic</b></td>
    <td><b>Papers</b></td>
  </tr>
  <tbody>
    {% for supclass in site.data.class_stats %}
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
    {% endfor %}
  </tbody>
</table>
</div>
</div>

{% include bib_search.liquid %}

<div class="publications">

{% bibliography %}

</div>
