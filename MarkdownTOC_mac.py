import sublime
import sublime_plugin
import re
import os.path
import pprint
from .bs4 import BeautifulSoup

# for dbug
pp = pprint.PrettyPrinter(indent=4)

pattern_reference_link = re.compile(r'\[.+?\]$') # [Heading][my-id]
pattern_link = re.compile(r'\[(.+?)\]\(.+?\)')  # [link](http://www.sample.com/)
pattern_ex_id = re.compile(r'\{#.+?\}$')         # [Heading]{#my-id}
pattern_tag = re.compile(r'<.*?>')
pattern_anchor = re.compile(r'<a\s+name="[^"]+"\s*>\s*</a>')
pattern_toc_tag_start = re.compile(r'<!-- *')
pattern_toc_tag_end = re.compile(r'-->')

pattern_h1_h2_equal_dash = "^.*?(?:(?:\r\n)|\n|\r)(?:-+|=+)$"

TOCTAG_START = "<!-- MarkdownTOC -->"
TOCTAG_END = "<!-- /MarkdownTOC -->"

class MarkdowntocInsert(sublime_plugin.TextCommand):

    def run(self, edit):

        if not self.find_tag_and_insert(edit):
            sels = self.view.sel()
            for sel in sels:
                attrs = self.get_settings()

                # add TOCTAG
                toc = TOCTAG_START + "\n"
                toc += "\n"
                toc += self.get_toc(attrs, sel.end(), edit)
                toc += "\n"
                toc += TOCTAG_END + "\n"

                self.view.insert(edit, sel.begin(), toc)
                self.log('inserted TOC')

        # TODO: process to add another toc when tag exists

    def get_toc_open_tag(self):
        search_results = self.view.find_all(
            "^<!-- MarkdownTOC .*-->\n",
            sublime.IGNORECASE)
        search_results = self.remove_items_in_codeblock(search_results)

        toc_open_tags = []
        for toc_open in search_results:
            if 0 < len(toc_open):

                toc_open_tag = {"region": toc_open}

                # settings in user settings
                settings_user = self.get_settings()

                # settings in tag
                tag_str = self.view.substr(toc_open)
                settings_tag = self.get_attibutes_from(tag_str)

                # merge
                toc_open_tag.update(settings_user)
                toc_open_tag.update(settings_tag)

                toc_open_tags.append(toc_open_tag)

        return toc_open_tags

    def get_toc_close_tag(self, start):
        close_tags = self.view.find_all("^" + TOCTAG_END + "\n")
        close_tags = self.remove_items_in_codeblock(close_tags)
        for close_tag in close_tags:
            if start < close_tag.begin():
                return close_tag

    def find_tag_and_insert(self, edit):
        """Search MarkdownTOC comments in document"""
        toc_starts = self.get_toc_open_tag()
        for dic in toc_starts:

            toc_start = dic["region"]
            if 0 < len(toc_start):

                toc_close = self.get_toc_close_tag(toc_start.end())

                if toc_close:
                    toc = self.get_toc(dic, toc_close.end(), edit)
                    tocRegion = sublime.Region(
                        toc_start.end(), toc_close.begin())
                    if toc:
                        self.view.replace(edit, tocRegion, "\n" + toc + "\n")
                        self.log('refresh TOC content')
                        return True
                    else:
                        self.view.replace(edit, tocRegion, "\n")
                        self.log('TOC is empty')
                        return False
        self.log('cannot find TOC tags')
        return False

    # TODO: add "end" parameter
    def get_toc(self, attrs, begin, edit):

        # Search headings in docment
        pattern_hash = "^#+?[^#]"
        headings = self.view.find_all(
            "%s|%s" % (pattern_h1_h2_equal_dash, pattern_hash))

        headings = self.remove_items_in_codeblock(headings)

        if len(headings) < 1:
            return ''

        items = []  # [[headingNum,text,position,anchor_id],...]
        for heading in headings:
            if begin < heading.end():
                lines = self.view.lines(heading)
                if len(lines) == 1:
                    # handle hash headings, ### chapter 1
                    r = sublime.Region(
                        heading.end() - 1, self.view.line(heading).end())
                    text = self.view.substr(r).strip().rstrip('#')
                    indent = heading.size() - 1
                    items.append([indent, text, heading.begin()])
                elif len(lines) == 2:
                    # handle - or + headings, Title 1==== section1----
                    text = self.view.substr(lines[0])
                    if text.strip():
                        indent = 1 if (
                            self.view.substr(lines[1])[0] == '=') else 2
                        items.append([indent, text, heading.begin()])

        if len(items) < 1:
            return ''

        # Shape TOC  ------------------
        items = format(items)

        # Depth limit  ------------------
        _depth = int(attrs['depth'])
        if 0 < _depth:
            items = list(filter((lambda i: i[0] <= _depth), items))

        # Create TOC  ------------------
        toc = ''
        _ids = []
        level_counters = [0]

        for item in items:
            _id = None
            _indent = item[0] - 1
            _text = item[1]
            _text = pattern_tag.sub('', _text) # remove html tags
            _text = _text.rstrip() # remove end space

            # Ignore links: e.g. '[link](http://sample.com/)' -> 'link'
            _text = pattern_link.sub('\\1', _text)

            # Add indent
            for i in range(_indent):
                _prefix = attrs['indent']
                # Support escaped characters like '\t'
                _prefix = _prefix.encode().decode('unicode-escape')
                toc += _prefix

            # Reference-style links: e.g. '# heading [my-anchor]'
            list_reference_link = list(pattern_reference_link.finditer(_text))

            # Markdown-Extra special attribute style: e.g. '# heading {#my-anchor}'
            match_ex_id = pattern_ex_id.search(_text)

            if len(list_reference_link):
                match = list_reference_link[-1]
                _text = _text[0:match.start()].replace('[','').replace(']','').rstrip()
                _id = match.group().replace('[','').replace(']','')
            elif match_ex_id:
                _text = _text[0:match_ex_id.start()].rstrip()
                _id = match_ex_id.group().replace('{#','').replace('}','')
            elif strtobool(attrs['autolink']):
                if strtobool(attrs['lowercase_only_ascii']):
                    # only ascii
                    _lower_text = ''.join(chr(ord(x)+('A'<=x<='Z')*32) for x in _text)
                else:
                    _lower_text = _text.lower()
                _id = self.replace_strings_in_id(_lower_text)
                _ids.append(_id)
                n = _ids.count(_id)
                if 1 < n:
                    _id += '-' + str(n-1)

            if attrs['style'] == 'unordered':
                list_prefix = '- '
            elif attrs['style'] == 'ordered':
                list_prefix = '1. '

            # escape brackets
            _text = _text\
                        .replace('(','\(')\
                        .replace(')','\)')\
                        .replace('[','\[')\
                        .replace(']','\]')

            if _id == None:
                toc += list_prefix + _text + '\n'
            elif attrs['bracket'] == 'round':
                toc += list_prefix + '[' + _text + '](#' + _id + ')\n'
            else:
                toc += list_prefix + '[' + _text + '][' + _id + ']\n'

            item.append(_id)

        self.update_anchors(edit, items, strtobool(attrs['autoanchor']))

        return toc

    def update_anchors(self, edit, items, autoanchor):
        """Inserts, updates or deletes a link anchor in the line before each header."""
        v = self.view
        # Iterate in reverse so that inserts don't affect the position
        for item in reversed(items):
            anchor_region_1 = v.line(item[2] - 1)  # -1 to get to previous line
            anchor_region_2 = v.line(item[2] - 2)  # -1 to get to previous line
            is_update = pattern_anchor.match(v.substr(anchor_region_1)) or pattern_anchor.match(v.substr(anchor_region_2))
            blank_line = ''.format(item[3])
            if autoanchor:
                if is_update:
                    # anchor_region = v.line(item[2] - 2)  # -1 to get to previous line
                    new_anchor = '<a name="{0}"></a>'.format(item[3])
                    
                    v.replace(edit, anchor_region_2, new_anchor)
                else:
                    # anchor_region = v.line(item[2] - 1)  # -1 to get to previous line
                    new_anchor = '\n<a name="{0}"></a>\n'.format(item[3])
                    # v.insert(edit, blank_line.end(), new_anchor)
                    v.insert(edit, anchor_region_1.end(), new_anchor)

    def get_setting(self, attr):
        settings = sublime.load_settings('MarkdownTOC.sublime-settings')
        return settings.get(attr)

    def get_settings(self):
        """return dict of settings"""
        return {
            "autoanchor":           self.get_setting('default_autoanchor'),
            "autolink":             self.get_setting('default_autolink'),
            "bracket":              self.get_setting('default_bracket'),
            "depth":                self.get_setting('default_depth'),
            "indent":               self.get_setting('default_indent'),
            "lowercase_only_ascii": self.get_setting('default_lowercase_only_ascii'),
            "style":                self.get_setting('default_style')
        }

    def get_attibutes_from(self, tag_str):
        """return dict of settings from tag_str"""

        tag_str_html = pattern_toc_tag_start.sub("<", tag_str)
        tag_str_html = pattern_toc_tag_start.sub(">", tag_str_html)

        soup = BeautifulSoup(tag_str_html)

        return soup.find('markdowntoc').attrs

    def remove_items_in_codeblock(self, items):

        codeblocks = self.view.find_all("^`{3,}[^`]*$")
        codeblockAreas = [] # [[area_begin, area_end], ..]
        i = 0
        while i < len(codeblocks)-1:
            area_begin = codeblocks[i].begin()
            area_end   = codeblocks[i+1].begin()
            if area_begin and area_end:
                codeblockAreas.append([area_begin, area_end])
            i += 2

        items = [h for h in items if is_out_of_areas(h.begin(), codeblockAreas)]
        return items

    def replace_strings_in_id(self, _str):
        replacements = self.get_setting('id_replacements')
        for _key in replacements:
            _substitute = _key
            _targets = replacements[_key]
            for _target in _targets:
                _str = _str.replace(_target, _substitute)
        return _str

    def log(self, arg):
        if self.get_setting('logging') == True:
            arg = str(arg)
            sublime.status_message(arg)
            pp.pprint(arg)

def is_out_of_areas(num, areas):
    for area in areas:
        if area[0] < num and num < area[1]:
            return False
    return True

def format(items):
    headings = []
    for item in items:
        headings.append(item[0])
    # --------------------------

    # minimize diff between headings -----
    _depths = list(set(headings)) # sort and unique
    # replace with depth rank
    for i, item in enumerate(headings):
        headings[i] = _depths.index(headings[i])+1
    # ----- /minimize diff between headings

    # --------------------------
    for i, item in enumerate(items):
        item[0] = headings[i]
    return items

def strtobool(val):
    """pick out from 'distutils.util' module"""
    if isinstance(val, str):
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError("invalid truth value %r" % (val,))
    else:
        return bool(val)


# Search and refresh if it's exist


class MarkdowntocUpdate(MarkdowntocInsert):

    def run(self, edit):
        MarkdowntocInsert.find_tag_and_insert(self, edit)


class AutoRunner(sublime_plugin.EventListener):

    def on_pre_save(self, view):
        # limit scope
        root, ext = os.path.splitext(view.file_name())
        ext = ext.lower()
        if ext in [".md", ".markdown", ".mdown", ".mdwn", ".mkdn", ".mkd", ".mark"]:
            view.run_command('markdowntoc_update')
