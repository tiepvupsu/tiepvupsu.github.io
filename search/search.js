$(document).ready(function() {

  // Initialize lunr with the fields to be searched, plus the boost.
  window.idx = lunr(function () {
    this.field('id');
    this.field('title');
    this.field('content', { boost: 10 });
    this.field('categories');
  });

  // Get the generated search_data.json file so lunr.js can search it locally.
  window.data = $.getJSON('/search/search_data.json?t=' + new Date());

  // Wait for the data to load and add it to lunr
  window.data.then(function(loaded_data){
    $.each(loaded_data, function(index, value){
      window.idx.add(
        $.extend({ "id": index }, value)
      );
    });

    // Display search result in GET string
    {
      let query = findGetParameter('keyword');
      if (query != null) {
        $("#search_box").val(query);
        search();
      }
    }
  });


  function findGetParameter(parameterName) {
      var result = null,
          tmp = [];
      location.search
      .substr(1)
          .split("&")
          .forEach(function (item) {
          tmp = item.split("=");
          if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
      });
      return result;
  }


  // Event when the form is submitted
  $("#site_search").submit(function(event){
      event.preventDefault();
      search();
  });

  function search() {
      var query = $("#search_box").val(); // Get the value for the text field
      var results = window.idx.search(query); // Get lunr to perform a search
      display_search_results(results); // Hand the results off to be displayed
  }


  function createExert(searchKey, content, numOfChars) {
      var searchWords = searchKey.split(" ")
      for (var i = 0; i < searchWords.length; i++) {
        searchWords[i] = searchWords[i].toLowerCase();
      }

      var exert = "";
      var begin = 0;
      var NUM_OF_CHARS_PER_SUBSTR = 100;
      var substr = "";
      var distance = false; // distance between two matched substring
      do {
        substr = content.substring(begin, begin + NUM_OF_CHARS_PER_SUBSTR).toLowerCase();
        var containSearchWord = false;
        for (var i = 0; i < searchWords.length; i++) {
          if (substr.indexOf(searchWords[i]) != -1) {
            containSearchWord = true;
            break;
          }
        }

        if (containSearchWord) {
          if (distance) exert += " .. ";
          exert += content.substring(begin, begin + NUM_OF_CHARS_PER_SUBSTR);
          distance = false;
        } else {
          distance = true;
        }

        begin += NUM_OF_CHARS_PER_SUBSTR; // next substring
        // console.log(substr.length);
      } while (substr.length != 0 && exert.length <= numOfChars);

      return exert;
  }

  function display_search_results(results) {
    var $search_results = $("#search_results");

    // Wait for data to load
    window.data.then(function(loaded_data) {

      // Are there any results?
      if (results.length) {
        $search_results.empty(); // Clear any old results

        // Iterate over the results
        results.forEach(function(result) {
          var item = loaded_data[result.ref];

          // Build a snippet of HTML for this result
          var appendString = '<li><h3><a class="result-title" href="' + item.url + '">' + item.title + '</a></h3>';
          appendString += '<p class="result-category"><b>Category: ' + item.categories + '</b></p>';
          var searchKeywords = $("#search_box").val();
          var exert = createExert(searchKeywords, item.content, 500);
          if (exert.length == 0) {
            exert = item.content.substring(0, 200);
          }
          appendString += '<p class="result-exert">' + exert + '...</p></li>';

          
          // Add the snippet to the collection of results.
          $search_results.append(appendString);

          // highlight keywords
          $(".result-exert").mark(searchKeywords, {separateWordSearch: true});
          $(".result-title").mark(searchKeywords, {separateWordSearch: true});
          $(".result-category").mark(searchKeywords, {separateWordSearch: true});
        });
      } else {
        // If there are no results, let the user know.
        $search_results.html('<li>No results found.<br/>Please check spelling, spacing,...</li>');
      }
    });
  }
});


