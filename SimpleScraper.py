import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_static_site():
    URL = "https://teestow.com/product-category/tvboo-merch/"
    try:
        response = requests.get(URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        merch = soup.find_all('li', class_="product")
        scraped_data = []

        for item in merch:
            title_tag = item.find('h2', class_='woocommerce-loop-product__title')
            title = title_tag.get_text(strip=True) if title_tag else "N/A"
            price_tag = item.find('span', class_='woocommerce-Price-amount')
            price = price_tag.get_text(strip=True) if price_tag else "N/A"
            scraped_data.append({
            'Title': title,
            'Price': price
            })

        print(f"Successfully scraped {len(scraped_data)} items.")
        return scraped_data
    
    except requests.exceptions.RequestException as e:

        print(f"Error during requests to {URL}: {e}")
        return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []

if __name__ == "__main__":
    data = scrape_static_site()

    if data:
        df = pd.DataFrame(data)
        df.to_csv('merch_data.csv', index=False)
        print("Data saved to merch_data.csv")

        print(df.head())