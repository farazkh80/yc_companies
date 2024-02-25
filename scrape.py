from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
import json
import time

def scrape_yc_companies():
    url = 'https://www.ycombinator.com/companies?batch=W24&batch=S23&batch=W23&batch=S22&batch=W22'

    # Selenium Chrome options
    options = Options()
    options.headless = True  # Run in headless mode
    service = Service('chromedriver')  # Replace with your ChromeDriver path

    # Start Selenium WebDriver
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait for new content to load
        time.sleep(1)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        print('Scrolling..., items found so far: ', len(driver.find_elements(By.XPATH, '//a[contains(@class, "_company_h0r20_")]')))

    # Get the page source after scrolling
    page_source = driver.page_source

    # Close the WebDriver
    driver.quit()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    company_pattern = re.compile(r'_company_h0r20_\d+')
    companies = soup.find_all('a', class_=company_pattern)
    company_data = []

    for company in companies:
        name = company.find('span', class_=re.compile(r'_coName_h0r20_\d+')).text if company.find('span', class_=re.compile(r'_coName_h0r20_\d+')) else None
        description = company.find('span', class_=re.compile(r'_coDescription_h0r20_\d+')).text if company.find('span', class_=re.compile(r'_coDescription_h0r20_\d+')) else None
        link = 'https://www.ycombinator.com' + company.get('href') if company.get('href') else None
        company_data.append({'name': name, 'description': description, 'link': link})

    return json.dumps(company_data, indent=4)

# Run the function and print the result
companies = scrape_yc_companies()


with open('yc_companies.json', 'w') as f:
    f.write(companies)