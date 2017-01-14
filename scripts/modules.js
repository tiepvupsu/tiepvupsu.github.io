---
---

{% include loadCSS.js %}
{% if site.debug %}{% include require.js %}{% else %}{% include require.min.js %}{% endif %}

var siteProperties = {
    url: "{{ site.url }}",
    baseurl: "{{ site.baseurl }}"
};

{% include main.js %}