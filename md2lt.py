import sys
import codecs
import re 
# from Unicode_VN import *


sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
sys.stdin = codecs.getreader('utf_8')(sys.stdin)

f = codecs.open('test.md', encoding='utf-8').read().splitlines();
# f = codecs.open('./_posts/2017-03-12-convexity.md', encoding='utf-8').read().splitlines();
i = -1

file = codecs.open("assets/latex/out.tex", "w", "utf-8")
# file.write()

############## TITLE #############33
first_line = "%!TEX root = book_CVX_SVM.tex"
file.write(first_line+'\n')



########### HEADERS ####################
HEADERS = [None, None, ' \\section', ' \\subsection', ' \\subsubsection', ' \\textbf']
c = 0 

in_math_mode = False
in_code_mode = False
in_fig_mode  = False 
in_tab_mode  = False
in_ital_mode = False 
in_bold_mode = False 
in_comment_mode = False 

BASE_URL = 'http://machinelearningcoban.com'


def myfind(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def convert_links(string):
	# print(a, b, c, d)
	id1 = string.find('[')
	if id1 == -1:
		# return string[:id1+1] + convert_links(string[id1+1:])
		return string 
	# find ]
	str2 = string[:id1]
	s2 = string[id1:]
	id2 = s2.find(']')
	if id2 == -1:
		return string[:id1+ id2 + 1] + convert_links(string[id1 + id2 +1:])

	if s2[id2 + 1] != '(':
		return string[:id1 + id2 + 2]+ convert_links(string[id1 + id2 + 2:])

	s3 = s2[id2 +1:]
	id3 = s3.find(')')
	if id3 == -1:
		return string[:id1 + id2 + id3 + 1] + convert_links(string[id1 + id2 + id3 + 1:])

	link = string[id1 + id2 +2: id1 + id2 + id3+1]
	display_name = string[id1 + 1: id1 + id2]
	if 'http' not in link:
		link = BASE_URL + link 

	return str2 + '\\href{' + link + '}{' + \
		display_name + '}' + convert_links(string[id1 + id2 + id3+2:])

str0 = 'alsfd `ksf` lkf `jksdf `'


def inlinecode(str0):
	id1 = str0.find('`')
	if id1 == -1:
		return str0 

	str1 = str0[id1 + 1:]
	id2 = str1.find('`')
	if id2 == -1:
		return str0 

	return str0[:id1] + '\\pythoninline{'+str0[id1+1:id1+id2+1] + '}' + \
			inlinecode(str0[id1+id2+2:])

print(inlinecode(str0))


add_title = False 
in_layout_mode = False
ignore_flag = True 
for line in f:
	# print(
	c += 1
	new_line = ' '+line.rstrip('\r') + ' '
	print(new_line)
	# if not in_layout_mode and '---' in new_line:
	# 	in_layout_mode = True 

	# ############# catch title 
	# if in_layout_mode:
	# 	if '---' in new_line:
	# 		in_layout_mode = False 

	if not add_title:
		if  'title' in new_line:
			add_title = True 
			# first : 
			id1 = new_line.index(':')
			str1 = new_line[id1+1:]
			#second :
			id2 = str1.index(':')
			id3 = str1[id2:].index('"')
			title = str1[id2 + 2:id2 + id3]
			print(id1, id2, id3, title)
			file.write('\\chapter{' +title + '}\n')
	else:
		if ignore_flag:
			if '---' in new_line:
				ignore_flag = False 
			else:
				continue 


	if ignore_flag:
		continue

	c += 1 
	# print(c)
	#### math 
	

	## in code mode 
	if '```' in new_line:
		in_code_mode = not in_code_mode 
		if in_code_mode: 
			new_line = ' \\begin{lstlisting}[language=Python]\r'
			file.write(new_line[1:]+'\n')
			continue 
		else: 
			new_line = ' \\end{lstlisting}\r'
			file.write(new_line[1:]+'\n')
			continue 
	if in_code_mode:
		file.write(new_line[1:]+'\n')
		continue 

	new_line = inlinecode(new_line)

	
	new_line = new_line.replace('\\_', '_')
	new_line = new_line.replace(' **', ' \\textbf{')
	new_line = new_line.replace('[**', '[\\textbf{')
	new_line = new_line.replace('(**', '(\\textbf{')
	new_line = new_line.replace('{**', '{\\textbf{')
	new_line = new_line.replace('** ', '} ')
	new_line = new_line.replace('**]', '}]')
	new_line = new_line.replace('**)', '})')
	new_line = new_line.replace('**}', '}}')
	new_line = new_line.replace('**.', '}.')
	new_line = new_line.replace('**,', '},')
	new_line = new_line.replace('**:', '}:')
	new_line = new_line.replace('**!', '}!')
	new_line = new_line.replace(' * ', '\t\\item ')
	new_line = new_line.replace(' *', ' \\textit{')
	new_line = new_line.replace('(*', '(\\textit{')
	new_line = new_line.replace('[*', '[\\textit{')
	new_line = new_line.replace('{*', '{\\textit{')
	new_line = new_line.replace('* ', '} ')
	new_line = new_line.replace('*)', '})')
	new_line = new_line.replace('*]', '}]')
	new_line = new_line.replace('*}', '}}')
	# new_line = new_line.replace('*}', '}}')
	new_line = new_line.replace('*,', '},')
	new_line = new_line.replace('*.', '}.')

	new_line = new_line.replace(' __', ' \\textbf{')
	new_line = new_line.replace('[__', '[\\textbf{')
	new_line = new_line.replace('(__', '(\\textbf{')
	new_line = new_line.replace('{__', '{\\textbf{')
	new_line = new_line.replace('__ ', '} ')
	new_line = new_line.replace('__]', '}]')
	new_line = new_line.replace('__)', '})')
	new_line = new_line.replace('__}', '}}')
	new_line = new_line.replace('__:', '}}')
	new_line = new_line.replace('__.', '}.')
	new_line = new_line.replace('__,', '},')

	new_line = new_line.replace(' _', ' \\textit{')
	new_line = new_line.replace('(_', '(\\textit{')
	new_line = new_line.replace('[_', '[\\textit{')
	new_line = new_line.replace('{_', '{\\textit{')
	new_line = new_line.replace('_ ', '} ')
	new_line = new_line.replace('_)', '})')
	new_line = new_line.replace('_]', '}]')
	new_line = new_line.replace('_}', '}}')
	new_line = new_line.replace('_}', '}}')
	new_line = new_line.replace('_:', '}:')
	new_line = new_line.replace('_,', '},')
	new_line = new_line.replace('_;', '};')
	new_line = new_line.replace('_.', '}.')

	new_line = new_line.replace('\\\(', '$')
	new_line = new_line.replace('\\\[', '\\begin{equation*}')
	new_line = new_line.replace('\\\]', '\\end{equation*}')
	new_line = new_line.replace('\\\|', '\\|')
	new_line = new_line.replace('\\\)', '$')

	## if comment line -> continue 
	if '<!--' in new_line:
		in_comment_mode = True 
		if '-->' in new_line:
			in_comment_mode = False 
		continue
	if in_comment_mode:
		if '-->' in new_line:
			in_comment_mode = False 
		else:
			continue

	if '<a name=' in new_line:
		continue

	

	#### Header 
	line = new_line
	if line[1] == '#' and not in_code_mode:
		h = 1
		while line[h] == '#':
			h += 1 

		levels = h -1
		# find next letter
		while line[h] in '0123456789. ':
			h += 1 

		new_line = HEADERS[min(levels,5)] + '{' + line[h:-1] + '}'
		# line = new_line 	
	new_line = convert_links(new_line)
	file.write(new_line[1:]+'\n')




# print(convert_links(string))