---
layout: post
title: Categories
---

<div class="tags-expo">
  <div class="tags-expo-list">
    {% for tag in site.categories %}
    <a href="#{{ tag[0] | slugify }}" class="post-tag">{{ tag[0] }}</a>
    <br>
    {% endfor %}
  </div>
  <hr/>
  <div class="tags-expo-section">
    {% for tag in site.categories %}
    <h2 id="{{ tag[0] | slugify }}">{{ tag[0] }}</h2>
    <ul class="tags-expo-posts">
      {% for post in tag[1] %}
      <!-- <li> -->
      <div>
        <span style="float: left;">
          <a href="{{ post.url }}">{{ post.title }}</a>
        </span>
       <span style="float: right;">
          {{ post.date | date_to_string }}
        </span>
        </div>
        <br>
      <!-- </li> -->
      <!-- </a> -->
      {% endfor %}
    </ul>
    {% endfor %}
  </div>
</div>

