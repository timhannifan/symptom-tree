import util 
import sys
import csv
import re
import bs4


BASE = 'https://icdcodelookup.com/icd-10/codes/'
# BASE_GEN = 'http://icd9.chrisendres.com/index.php?srchtype=diseases&srchtext='
BASE_GEN = 'https://www.icd10data.com/search?s='
COUNT = 0



def translate_code(code):

	url = BASE + str(code)
	ro = util.get_request(url)
	html = util.read_request(ro)
	soup = bs4.BeautifulSoup(html, "html5lib")

	rv = soup.find('h3', class_='bold')

	if not rv:
		print(code)
	else:
		rv = rv.get_text().strip()

	# lst = soup.find('div', class_="codeSearchContainer").find_all('script')
	# txt = lst[2].get_text() 
	# print(txt)
	# rv = re.search(r'pageTitle = "(.*?)ICD', txt).group(1).strip()[8:]

	return rv

def translate_general(code):

	# url = BASE_GEN + str(code) + '&Submit=Search&action=search'
	url = BASE_GEN + str(code) + '&codebook=icd9volume1'
	ro = util.get_request(url)
	html = util.read_request(ro)
	soup = bs4.BeautifulSoup(html, "html5lib")
	# txt = soup.find('div', class_='dlvl')

	rv = soup.find('div').next_sibling.next_sibling.find('div', class_='searchPadded')

	if not rv:
		print(code)
	else:
		rv = rv.text

	return rv


# top['DIAG1'] = top['DIAG1'].apply(lambda x: get_ICD.translate_code(x))
# top['DIAG11'] = top['DIAG1'].apply(lambda x: x.strip('-'))
# top.drop(['PREGNANT','INJURY','USETOBAC','MAJOR'], axis=1, inplace=True) 
