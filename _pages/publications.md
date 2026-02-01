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

<div class="venue-stats-table">
<table>
  <thead>
    <tr>
      <th><b>Venue</b></th>
      <th><b>Papers</b></th>
      <th><b><i class="fa-solid fa-handshake"></i> / <i class="fa-solid fa-envelope"></i></b></th>
    </tr>
  </thead>
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

{% include bib_search.liquid %}

<div class="publications">

{% bibliography %}

</div>
