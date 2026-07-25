---
layout: default
title: Home
---

<section class="hero">
  <p class="eyebrow">Public notebook</p>
  <h1>an Euclidean space</h1>
  <p class="hero-copy">
    Detailed notes on mathematics, analysis, optimization, and the ideas that connect them.
    Written to preserve proofs, not just conclusions.
  </p>
  <div class="hero-actions">
    <a class="button button-primary" href="{{ '/notes/' | relative_url }}">Browse the notes</a>
    <a class="text-link" href="{{ '/about/' | relative_url }}">About this site</a>
  </div>
</section>

<section class="section-block">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Archive</p>
      <h2>Selected notes</h2>
    </div>
    <a class="text-link" href="{{ '/notes/' | relative_url }}">View all notes <span aria-hidden="true">→</span></a>
  </div>

  <div class="home-note-list">
    {% assign featured_notes = site.notes | sort: "source_title" %}
    {% for note in featured_notes limit: 6 %}
      <article class="home-note">
        <div class="home-note-meta">
          <span>{{ note.course }}</span>
          <span>{{ note.sequence }}</span>
        </div>
        <h3><a href="{{ note.url | relative_url }}">{{ note.title }}</a></h3>
        <p>{{ note.excerpt | strip_html | normalize_whitespace | truncate: 190 }}</p>
      </article>
    {% endfor %}
  </div>
</section>
