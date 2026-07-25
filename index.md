---
layout: default
title: Home
---

<section class="hero">
  <p class="eyebrow">A notebook in public</p>
  <h1>an Euclidean space</h1>
  <p class="hero-copy">
    Detailed notes on mathematics, analysis, optimization, and the ideas that connect them.
    Written to preserve proofs, not just conclusions.
  </p>
  <div class="hero-actions">
    <a class="button button-primary" href="{{ '/notes/' | relative_url }}">Browse the notes</a>
    <a class="button button-secondary" href="{{ '/about/' | relative_url }}">About this site</a>
  </div>
</section>

{% assign course_groups = site.notes | group_by: "course" %}

<section class="metrics" aria-label="Archive summary">
  <div class="metric">
    <strong>{{ site.notes | size }}</strong>
    <span>published notes</span>
  </div>
  <div class="metric">
    <strong>{{ course_groups | size }}</strong>
    <span>course threads</span>
  </div>
  <div class="metric">
    <strong>∞</strong>
    <span>room to continue</span>
  </div>
</section>

<section class="section-block">
  <div class="section-heading">
    <div>
      <p class="eyebrow">The archive</p>
      <h2>Published notes</h2>
    </div>
    <a class="text-link" href="{{ '/notes/' | relative_url }}">View all notes <span aria-hidden="true">→</span></a>
  </div>

  <div class="note-grid">
    {% assign featured_notes = site.notes | sort: "source_title" %}
    {% for note in featured_notes limit: 6 %}
      <article class="note-card">
        <div class="note-card-meta">
          <span>{{ note.course }}</span>
          <span>{{ note.sequence }}</span>
        </div>
        <h3><a href="{{ note.url | relative_url }}">{{ note.title }}</a></h3>
        <p>{{ note.excerpt | strip_html | normalize_whitespace | truncate: 176 }}</p>
        <a class="card-link" href="{{ note.url | relative_url }}" aria-label="Read {{ note.title }}">Read note <span aria-hidden="true">→</span></a>
      </article>
    {% endfor %}
  </div>
</section>

<section class="principle">
  <p class="eyebrow">Working principle</p>
  <blockquote>
    A proof is not a compressed answer. It is a visible chain of obligations.
  </blockquote>
  <p>
    These notes retain the definitions, derivations, counterexamples, and boundary conditions
    needed to reconstruct an argument later.
  </p>
</section>
