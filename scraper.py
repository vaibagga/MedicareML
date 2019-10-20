from bs4 import BeautifulSoup
import requests

url = "https://www.lalpathlabs.com/test-for-diabetes"



with requests.get(url) as html_file:
    print(html_file.content)
    soup=BeautifulSoup(html_file.content, 'html5lib')
    print(soup.find_all("div", class_ = "greyBox Allergy"))

