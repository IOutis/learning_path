from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

def scrape_url(website: str) -> str:
    """Scrapes any webpage given as a parameter and returns raw HTML data."""
    with sync_playwright() as p:
        # Launch a browser (headless by default)
        browser = p.chromium.launch()
        
        # Open a new browser page
        page = browser.new_page()
        
        # Navigate to the website
        page.goto(website)
        print("Page loaded...")
        
        # Get the page content
        html = page.content()
        
        # Close the browser
        browser.close()
        
        return html

def extract_data(html: str) -> pd.DataFrame:
    """Extracts and returns a DataFrame with the required attributes."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Find the table containing the data
    table = soup.find("table", {"id": "cphBody_GridPriceData"})  # Update the ID based on the actual table ID
    
    if not table:
        raise ValueError("Table not found on the page.")
    
    # Extract table headers
    headers = [th.text.strip() for th in table.find("tr").find_all("th")]
    
    # Extract table rows
    rows = []
    for tr in table.find_all("tr")[1:]:  # Skip the header row
        cells = [td.text.strip() for td in tr.find_all("td")]
        rows.append(cells)
    
    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df

def filter_data(df: pd.DataFrame, commodity: str = None, state: str = None, date_range: tuple = None) -> pd.DataFrame:
    """Filters the DataFrame based on commodity, state, and date range."""
    if commodity:
        df = df[df["Commodity"] == commodity]
    if state:
        df = df[df["State"] == state]
    if date_range:
        start_date, end_date = date_range
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    return df

def main():
    # URL of the Agmarknet page
    url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    
    # Scrape the webpage
    html = scrape_url(url)
    
    # Extract data from the HTML
    df = extract_data(html)
    
    # Filter data (optional)
    commodity = "Tomato"  # Replace with the desired commodity
    state = "Telangana"   # Replace with the desired state
    date_range = (datetime(2023, 10, 1), datetime(2023, 10, 31))  # Replace with the desired date range
    
    filtered_df = filter_data(df, commodity=commodity, state=state, date_range=date_range)
    
    # Save the filtered data to a CSV file
    filtered_df.to_csv("agmarknet_data.csv", index=False)
    print("Data saved to agmarknet_data.csv")

if __name__ == "__main__":
    main()