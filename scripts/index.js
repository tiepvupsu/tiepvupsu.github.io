"use strict";

require(["jquery", "jquery-ui"], function($) {
    $(function() {
        $("#tabs").tabs({
            beforeActivate: function(event, ui) {
                var category = ui.newPanel.attr('id').substring(5);
                var selectedTag = ui.newPanel.data("selectedTag");
                var newLocation = "#" + category;
                if (selectedTag) {
                    newLocation += ";" + selectedTag;
                }
                window.location = newLocation;
            }
        });
        $("#tabs .ui-tabs-panel").each(function() {
            var category = $(this).attr("id").substring(5);
            var tagCloud = $(this).find("ul.tag-box");
            var postList = $(this).find("ul.post-list");
            var tags = {};
            postList.find("meta[itemprop='keywords']").each(function() {
                $.each($(this).attr("content").split(","), function(index, tag) {
                    if (tags[tag]) {
                        tags[tag]++;
                    } else {
                        tags[tag] = 1;
                    }
                });
            });
            $(this).bind("tagSelected", function(event, tag) {
                postList.find("meta[itemprop='keywords']").each(function() {
                    var li = $(this).parent().parent();
                    var matches = tag && $(this).attr("content").split(",").indexOf(tag) != -1;
                    if (matches != $(li).is(":visible")) {
                        if (matches) {
                            $(li).show("slow");
                        } else {
                            $(li).hide("slow");
                        }
                    }
                });
            });
            $.each(tags, function(tag, count) {
                var li = $("<li></li>");
                li.appendTo(tagCloud);
                var a = $("<a href=''>" + tag + " <span>" + count + "</span></a>");
                a.appendTo(li);
                // TODO: we might use a simple href as well, but we will support multiple selections later
                a.click(function(e) {
                    e.preventDefault();
                    window.location = "#" + category + ";" + tag;
                });
            });
            postList.find("li").hide();
        });
        $("#tabs").removeClass("hidden");
        var processHash = function() {
            var hash = window.location.hash;
            if (hash && hash.substring(0, 1) == "#") {
                var parts = hash.substring(1).split(";");
                if (parts.length == 1 || parts.length == 2) {
                    var selectedCategory = parts[0];
                    var selectedTag = parts.length == 2 ? parts[1] : null;
                    var index = 0;
                    $("#tabs .ui-tabs-panel").each(function() {
                        if ($(this).attr("id") == "tabs-" + selectedCategory) {
                            $(this).data("selectedTag", selectedTag);
                            $("#tabs").tabs("option", "active", index);
                            $(this).trigger("tagSelected", [selectedTag]);
                        }
                        index++;
                    });
                }
            }
        };
        processHash();
        $(window).bind("hashchange", function(e) {
            processHash();
        });
    });
});
