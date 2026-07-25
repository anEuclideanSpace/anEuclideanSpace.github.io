---
layout: default
title: Notes
permalink: /notes/
---

<section class="page-intro">
  <p class="eyebrow">Knowledge archive</p>
  <h1>Notes</h1>
  <p>
    Long-form mathematical and technical notes, organized by source thread.
    Each page is meant to stand on its own while preserving the proofs and conceptual dependencies.
  </p>
</section>

{% assign course_groups = site.notes | group_by: "course" | sort: "name" %}

<div class="archive">
  {% for group in course_groups %}
    <section class="archive-group">
      <header class="archive-header">
        <p class="eyebrow">Course thread</p>
        <h2>{{ group.name }}</h2>
        <span>{{ group.items | size }} {% if group.items.size == 1 %}note{% else %}notes{% endif %}</span>
      </header>

      <div class="archive-list">
        {% assign ordered_notes = group.items | sort: "sequence" %}
        {% for note in ordered_notes %}
          <article class="archive-item">
            <div class="archive-sequence">{{ note.sequence }}</div>
            <div>
              <h3><a href="{{ note.url | relative_url }}">{{ note.title }}</a></h3>
              <p>{{ note.excerpt | strip_html | normalize_whitespace | truncate: 220 }}</p>
              {% if note.tags %}
                <div class="tag-list" aria-label="Topics">
                  {% for tag in note.tags limit: 4 %}
                    <span>{{ tag }}</span>
                  {% endfor %}
                </div>
              {% endif %}
            </div>
          </article>
        {% endfor %}
      </div>
    </section>
  {% endfor %}
</div>
