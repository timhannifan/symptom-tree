'''
Scrapes ICD code data from website, creates a dicitonary of code to long
form string, exports dictionary to json file.
'''

import util
import bs4

BASE_GEN = 'https://www.icd10data.com/search?s='
BAD_DIAGNOSES = ["V990", "V991", "V992", "V997", "-9", "V99", "V97"]

def translate_general(code):
    '''
    Given an IDC-code, scrapes www.icd10data.com and returns a meaningful
    translation as a string

    Input:
        code (int): an IDC-code
    Output:
        rv (string): translation of the IDC-code
    '''
    url = BASE_GEN + str(code) + '&codebook=icd9volume1'
    ro = util.get_request(url)
    html = util.read_request(ro)
    soup = bs4.BeautifulSoup(html, "html5lib")
    rv = None
    search = soup.find('div').next_sibling.next_sibling.find('div',
        class_='searchPadded')

    if search and search.text:
        rv = search.text

    return rv

def make_dictionary(df):
    '''
    Returns a dictionary with ICD-codes and their translations as key
    value pairs and writes the dictionary to a JSON file.

    Input:
        df (dataframe): a dataframe with column of ICD-codes
    Output:
        icd_dictionary (dict): dictionary with code, translation pairs
    '''
    icd_dictionary = {}
    df.query("DIAGNOSIS_SHORT_1 not in @BAD_DIAGNOSES", inplace=True)
    new_df = df['DIAGNOSIS_SHORT_1'].to_frame()
    new_df = new_df.dropna(subset=['DIAGNOSIS_SHORT_1'])

    for row in new_df.itertuples(index=True, name='Pandas'):
        code = getattr(row, "DIAGNOSIS_SHORT_1")
        if code:
            if code not in icd_dictionary.keys():
                icd_dictionary[code] = translate_general(code)

    with open('data/icd_codes.json', 'w') as fp: 
        json.dump(icd_dictionary, fp, sort_keys=True, indent=4) 

    return icd_dictionary
