# ==============================================================================
# 0. INSTALL DEPENDENCIES (IF NOT ALREADY INSTALLED)
# ==============================================================================
# Run this command in your terminal/command prompt if you haven't installed them:
# pip install selenium pandas openpyxl

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import os

# ==============================================================================
# 1. Configuration and Setup
# ==============================================================================

# --- Path to your Excel file containing tickers ---
excel_file_path = 'Yahoo_Financials_BK_UPDATED.xlsx'

# --- Column name in your Excel file that contains the stock tickers ---
ticker_column_name = 'Ticker'

# --- Base URL for SETTrade financial statements ---
base_url = "https://www.settrade.com/th/equities/quote/{ticker}/financial-statement/full"

# --- Output file for combined data ---
combined_output_file = 'Yahoo_Financials_BK_2024_Specific.xlsx'

# Initialize a list to store all dataframes
all_data = []

# Setup Chrome options for debugging
options = webdriver.ChromeOptions()
# Non-headless mode for debugging
# options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--log-level=3')
options.add_argument('--incognito')

# Initialize the driver
driver = None
try:
    print("🚀 Starting a local Chrome browser...")
    driver = webdriver.Chrome(options=options)
    print("✅ Browser started successfully.")
except WebDriverException as e:
    print(f"❌ Error starting Chrome browser: {e}")
    exit()

# ==============================================================================
# 2. Read Tickers from Excel
# ==============================================================================
try:
    df_tickers = pd.read_excel(excel_file_path)
    if ticker_column_name not in df_tickers.columns:
        raise ValueError(f"❌ Column '{ticker_column_name}' not found in '{excel_file_path}'.")
    tickers = df_tickers[ticker_column_name].dropna().unique().tolist()
    print(f"📄 Found {len(tickers)} tickers to process from '{excel_file_path}'.")
except FileNotFoundError:
    print(f"❌ Error: Excel file '{excel_file_path}' not found.")
    if driver: driver.quit()
    exit()
except ValueError as e:
    print(f"❌ Error reading Excel file: {e}")
    if driver: driver.quit()
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while reading the Excel file: {e}")
    if driver: driver.quit()
    exit()

# ==============================================================================
# 3. Loop, Scrape, Process, and Save Data for Each Ticker
# ==============================================================================
try:
    for i, ticker in enumerate(tickers):
        ticker_str = str(ticker).strip().upper()
        if not ticker_str:
            print(f"⚠️ Skipping empty ticker at index {i}.")
            continue

        current_url = base_url.format(ticker=ticker_str)
        print(f"\n--- Processing Ticker {i+1}/{len(tickers)}: {ticker_str} ---")
        print(f"🔗 Navigating to: {current_url}")

        try:
            driver.get(current_url)
            time.sleep(2)  # Initial pause for page load

            # Wait for the caret (dropdown toggle) to be clickable
            print("⏳ Waiting for the caret dropdown toggle to load...")
            caret_selector = "div.caret"
            wait = WebDriverWait(driver, 30)
            try:
                caret = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, caret_selector)))
                print("✅ Caret found! Clicking to open dropdown...")
                caret.click()
                time.sleep(1)  # Pause for dropdown to open
            except TimeoutException:
                print(f"❌ Timeout waiting for caret '{caret_selector}'. Trying fallback selector...")
                # Fallback: parent div of dropdown
                caret_selector = "div.dropdown-toggle"
                caret = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, caret_selector)))
                print("✅ Fallback caret found! Clicking to open dropdown...")
                caret.click()
                time.sleep(1)

            # Find and click the "2024 - Budget Year" option
            print("🔄 Selecting '2024 - Budget Year' from dropdown...")
            year_option_selector = "//font[contains(text(), '2024 - Budget Year')]"
            try:
                year_option = wait.until(EC.element_to_be_clickable((By.XPATH, year_option_selector)))
                print("✅ Found '2024 - Budget Year' option!")
                year_option.click()
                time.sleep(2)  # Pause for selection to process
            except TimeoutException:
                print(f"❌ Timeout waiting for '2024 - Budget Year'. Printing available options...")
                # Print all font elements in dropdown for debugging
                dropdown_options = driver.find_elements(By.CSS_SELECTOR, "div.dropdown-menu font")
                options_text = [opt.text.strip() for opt in dropdown_options if opt.text.strip()]
                print(f"🔍 Available dropdown options: {options_text}")
                continue

            # Click the search button
            print("🔍 Clicking the search button...")
            search_button_selector = "button.btn.btn-md.fs-20px.text-white.bg-primary.search-btn"
            try:
                search_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, search_button_selector)))
                search_button.click()
                print("✅ Search button clicked!")
            except TimeoutException:
                print(f"❌ Timeout waiting for search button '{search_button_selector}'.")
                continue
            time.sleep(5)  # Wait for data to load

            # Extract data from specified divs
            print("⏳ Waiting for the financial data section to load...")
            data_container_selector = "div.table-simple-customfield"
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, data_container_selector)))
            print("✅ Data container found!")

            # Initialize lists for metrics and values
            metrics = []
            values = []
            capture_data = False

            # Find metric and value divs
            metric_divs = driver.find_elements(By.CSS_SELECTOR, "div.col-6.ps-5")
            value_divs = driver.find_elements(By.CSS_SELECTOR, "div.col-6.text-end")

            print(f"🔍 Found {len(metric_divs)} metric divs and {len(value_divs)} value divs.")

            # Iterate over paired divs
            for metric_div, value_div in zip(metric_divs, value_divs):
                metric_text = metric_div.text.strip()
                value_text = value_div.text.strip()

                # Start capturing at "Cash and cash equivalents"
                if metric_text == "Cash and cash equivalents":
                    capture_data = True

                if capture_data and value_text:
                    metrics.append(metric_text)
                    values.append(value_text)

                # Stop capturing after "Total liabilities and shareholders' equity"
                if metric_text == "Total liabilities and shareholders' equity":
                    break

            if not metrics:
                print(f"❌ No data found between 'Cash and cash equivalents' and 'Total liabilities and shareholders' equity' for {ticker_str}. Skipping.")
                continue

            # Create DataFrame
            df = pd.DataFrame({
                'Ticker': ticker_str,
                'Financial_Metric': metrics,
                'Value_2024': values
            })

            # Convert values to numeric
            df['Value_2024'] = pd.to_numeric(df['Value_2024'].str.replace(',', '').str.replace('−', '-').str.strip(), errors='coerce')

            print(f"📊 Data for {ticker_str} extracted successfully.")
            print(f"🔍 DataFrame shape: {df.shape}")
            print(f"🔍 First few rows:\n{df.head()}")

            # Add to combined data
            all_data.append(df)
            print(f"✅ Added {ticker_str} data to combined dataset.")

        except TimeoutException as e:
            print(f"❌ Timeout waiting for page elements for {ticker_str}: {e}")
        except WebDriverException as e:
            print(f"❌ Selenium WebDriver error for {ticker_str}: {e}")
        except Exception as e:
            print(f"❌ An error occurred while processing {ticker_str}: {e}")
            import traceback
            print(f"🔍 Detailed error traceback:\n{traceback.format_exc()}")

        time.sleep(5)  # Pause to avoid server overload

except Exception as main_e:
    print(f"An unhandled error occurred during the main scraping process: {main_e}")
    import traceback
    print(f"🔍 Detailed error traceback:\n{traceback.format_exc()}")

finally:
    # Combine and save data
    if all_data:
        print(f"\n📊 Combining data from {len(all_data)} tickers...")
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_excel(combined_output_file, index=False, header=True, sheet_name='Financial Data 2024')
        print(f"💾 Successfully saved combined financial data to '{combined_output_file}'")
        print(f"📊 Total rows in combined file: {len(combined_df)}")
        print(f"📊 Total columns in combined file: {len(combined_df.columns)}")
    else:
        print("⚠️ No data was successfully scraped. No combined file created.")

    # Close browser
    if driver:
        print("\n👋 Closing the browser.")
        driver.quit()
    print("\n--- Script Finished ---")