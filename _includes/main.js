window.GoogleAnalyticsObject = "__ga__";
window.__ga__ = function() {
    for (var i=0; i<arguments.length; i++) {
        var arg = arguments[i];
        if (arg.constructor == Object && arg.hitCallback) {
            arg.hitCallback();
        }
    }
};
window.__ga__.q = [["create", "UA-89509207-1", {"cookieDomain": "auto", "siteSpeedSampleRate": 10}],
                   ["set", "forceSSL", true]];
window.__ga__.l = Date.now();

require.config({
    baseUrl: "/scripts",
    paths: {
        "jquery": [
            "//ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min",
            "jquery.min"
        ],
        "jquery-ui": [
            "//ajax.googleapis.com/ajax/libs/jqueryui/1.11.2/jquery-ui.min",
            "jquery-ui.min"
        ],
        "chart": "Chart.min",
        "ga": localStorage.dontTrack ? "data:application/javascript," : [
            "//www.google-analytics.com/analytics",
            "data:application/javascript,"
        ],
    },
    shim: {
        "ga": {
            exports: "__ga__"
        },
        "prism": {
            exports: "Prism"
        }
    }
});

require(["jquery"], function($) {
    $.each(pageProperties.scripts, function(index, script) {
        require([script]);
    });
    $(function() {
        if ($("code[class*='language-'], [class*='language-'] code, code[class*='lang-'], [class*='lang-'] code").length) {
            require(["prism"], function(prism) {
                prism.highlightAll();
            });
            loadCSS(siteProperties.baseurl + "/style/prism.css");
        }
        $("article a").each(function() {
            var url = $(this).attr("href");
            if ((url.indexOf("http://") == 0 || url.indexOf("https://") == 0) && url.indexOf(siteProperties.url + "/") != 0) {
                $(this).click(function(e) {
                    require(["ga"], function(ga) {
                        ga("send", "event", "outbound", "click", url, {"hitCallback":
                            function () {
                                document.location = url;
                            }
                        });
                    });
                    e.preventDefault();
                });
            }
        });
    });
});

require(["ga"], function(ga) {
    if (pageProperties.category) {
        ga("set", "dimension1", pageProperties.category);
    }
    ga("send", "pageview", {
      "page": pageProperties.url,
      "title": pageProperties.title
    });
});

if (typeof disqus_identifer !== "undefined") {
    require(["//veithen.disqus.com/embed.js"]);
}
