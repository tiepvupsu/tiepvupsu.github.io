"""
1. go to disqus.com/ Admin / Community  Migration Tools / Start URL mapper
2. Download the CSV file by looking for "you can download a CSV file here"
3. Extract the downloaded file, rename it to disqus_links.csv and put it 
in the top folder of the blog 
4. run this file, copy results back to disqus_links.csv. Remember to 
remove the last line [Finised in 0.0s] :D
5. Go back to the link in step 1, up load this file in "Choose file"
6. Done and wait 
"""

import csv 

MY_URL = 'https://tiepvupsu.github.io'
def process_link(link):
	# find all locations of '/'
	slash_id = [n for n in xrange(len(line[0])) if line[0].find('/', n) == n]
	print(slash_id)
	old_path = link[:slash_id[2]]
	if old_path != MY_URL:
		print(link + ',' + (65 - len(link))*' ' + MY_URL + link[slash_id[2]:])
	else: 
		print(link)

with open('disqus_links.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter = ',')
	for line in csvreader:
		process_link(line[0])
			 
