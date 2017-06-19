---
layout: post
title: Post Search
permalink: /search/
---

<div class="container">
    <form action="get" id="site_search">
        <div class="input-group">
            <input id="search_box" type="text" placeholder="type here and enter" />
            <button type="submit" class="btn btn-default">
                <i class="fa fa-search" aria-hidden="true"></i>
            </button>
        </div>
        <ul id="search_results"></ul>
    </form>
</div>

<script src="lunr.min.js"></script>
<script src="search.js"></script>


