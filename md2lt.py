import sys
import codecs
import re 
# from Unicode_VN import *


sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
sys.stdin = codecs.getreader('utf_8')(sys.stdin)

f = codecs.open('test.md', encoding='utf-8')
i = -1

file = codecs.open("out.tex", "w", "utf-8")
# file.write()

########### HEADERS ####################
HEADERS = [None, None, '\\section', '\\subsection', '\\subsubsection', '\\textbf']
c = 0 

in_math_mode = False
in_code_mode = False
in_fig_mode  = False
in_tab_mode  = False
in_ital_mode = False 
in_bold_mode = False 
in_comment_mode = False 

for line in f:
	c += 1 
	print(c)
	#### math 
	new_line = line
	new_line = new_line.replace('\\\(', '$')
	new_line = new_line.replace('\\\[', '$$')
	new_line = new_line.replace('\\\]', '$$')
	new_line = new_line.replace('\\\)', '$')
	new_line = new_line.replace('\\_', '_')

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

	## in code mode 
	if '```' in new_line:
		in_code_mode = not in_code_mode 
		if in_code_mode: 
			new_line = '\\begin{listing}\n'
		else: 
			new_line = '\\end{listing}\n'

	#### Header 
	if line[0] == '#' and not in_code_mode:
		h = 1
		while line[h] == '#':
			h += 1 

		levels = h 
		# find next letter
		while line[h] in '0123456789. ':
			h += 1 

		new_line = HEADERS[min(levels,5)] + '{' + line[h:-1] + '}'
		# line = new_line 

	## check italic and bold 
	line = new_line
	new_line2 = ''
	ii = -1 
	flag = False 
	while ii < len(line) - 1:
		ii += 1 
		if line[ii] == '$':
			in_math_mode = not in_math_mode 
			if line[ii+1] == '$':
				ii += 1 
			print('in math mode' + str(in_math_mode))

		if not in_math_mode and not in_code_mode and line[ii] == '_':
			flag = True 
			if line[ii+1] == '_':
				ii += 1 
				in_bold_mode = not in_bold_mode 
				ii+= 1 
				if in_bold_mode:
					new_line2 += '\\textbf{'
				else:
					new_line2 += '}'
			else: 	
				in_ital_mode = not in_ital_mode 
				ii += 1
				if in_ital_mode: 
					new_line2 += '\\textit{'
				else: 
					new_line2 += '}'
		if not in_math_mode and not in_code_mode and line[ii] == '*':
			flag = True
			if line[ii+1] == '*':
				ii += 1 
				in_bold_mode = not in_bold_mode 
				ii+= 1 
				if in_bold_mode:
					new_line2 += '\\textbf{'
				else:
					new_line2 += '}'
			else: 	
				in_ital_mode = not in_ital_mode 
				ii += 1
				if in_ital_mode: 
					new_line2 += '\\textit{'
				else: 
					new_line2 += '}'
			### links 
			


		new_line2 += line[ii]





	if not flag:
		myline = new_line
	else:
		myline = new_line2

	file.write(myline)
