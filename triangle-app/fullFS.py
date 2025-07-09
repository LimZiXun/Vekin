import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
import numpy as np
import time
import os
import random
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. LOAD COMPANY SYMBOLS FROM CSV
# ==============================================================================
def load_company_symbols(csv_file_path):
    """Load company symbols from the provided CSV file"""
    print(f"📁 Loading company list from: {csv_file_path}")
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        header_line_index = -1
        for i, line in enumerate(lines):
            if 'Symbol,Company,Market' in line:
                header_line_index = i
                break
        
        if header_line_index == -1:
            raise ValueError("Could not find proper header row in CSV")
        
        df = pd.read_csv(csv_file_path, skiprows=header_line_index)
        symbols = df['Symbol'].dropna().astype(str).str.strip()
        symbols = symbols[symbols != ''].unique().tolist()
        
        print(f"✅ Loaded {len(symbols)} company symbols")
        print(f"📊 First 10 symbols: {symbols[:10]}")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return []

# ==============================================================================
# 2. UTILITY FUNCTION FOR STANDARDIZING NULL VALUES
# ==============================================================================
def standardize_null_values(df):
    """Convert only explicit null-like values to pd.NA while preserving valid financial data"""
    # Define explicit null-like values to convert to pd.NA
    null_values = [
        'nan', 'NaN', 'NAN', 'null', 'NULL', 'Null', 
        'none', 'None', 'NONE', 'n/a', 'N/A', 'n.a.', 'N.A.',
        'na', 'NA', '#N/A', '#NA', '#NULL!', '#DIV/0!', '#VALUE!',
        'not available', 'NOT AVAILABLE', 'not applicable', 'NOT APPLICABLE',
        'undefined', 'UNDEFINED', 'missing', 'MISSING', 'blank', 'BLANK',
        'empty', 'EMPTY', 'no data', 'NO DATA'
    ]
    
    for col in df.columns:
        if col in ['Symbol', 'Financial_Metric']:
            # For string columns, only convert empty strings or exact null values
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['', *null_values], pd.NA)
        else:
            # For Value column, preprocess to handle commas and negative numbers
            df[col] = df[col].astype(str).str.strip()
            # Replace null-like values with pd.NA
            df[col] = df[col].replace(null_values, pd.NA)
            # Preprocess: remove commas and handle negative numbers
            df[col] = df[col].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)
            # Convert to numeric, preserving original strings if conversion fails
            df[col] = df[col].apply(
                lambda x: pd.to_numeric(x, errors='coerce') if not pd.isna(x) else pd.NA
            )
            # If numeric conversion fails, revert to original string (before comma removal)
            original_values = df[col].copy()
            df[col] = df[col].where(df[col].notna(), original_values)
    
    return df

# ==============================================================================
# 3. BROWSER MANAGEMENT CLASS
# ==============================================================================
class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
    
    async def initialize(self):
        """Initialize browser with proper error handling"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding'
                ]
            )
            
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                ignore_https_errors=True
            )
            
            self.page = await self.context.new_page()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            await self.cleanup()
            return False
    
    async def cleanup(self):
        """Safely cleanup browser resources"""
        try:
            if self.page:
                await self.page.close()
                self.page = None
        except Exception as e:
            logger.warning(f"Error closing page: {e}")
        
        try:
            if self.context:
                await self.context.close()
                self.context = None
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        
        try:
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
    
    async def restart_browser(self):
        """Restart browser if it becomes unresponsive"""
        logger.info("Restarting browser...")
        await self.cleanup()
        await asyncio.sleep(3)
        return await self.initialize()

# ==============================================================================
# 4. MODIFIED SCRAPE SINGLE COMPANY FUNCTION
# ==============================================================================
async def scrape_company_financial_data(browser_manager, symbol, max_retries=3):
    """Scrape financial data for a single company from div-based structure for 2567 - งบปี"""
    
    for attempt in range(max_retries):
        try:
            if not browser_manager.page or browser_manager.page.is_closed():
                logger.warning(f"Browser page is closed, restarting for {symbol}")
                if not await browser_manager.restart_browser():
                    raise Exception("Failed to restart browser")
            
            url = f"https://www.settrade.com/th/equities/quote/{symbol}/financial-statement/full"
            print(f"🔗 [{symbol}] Navigating to financial page (Attempt {attempt + 1}/{max_retries})")
            
            await browser_manager.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            print(f"✅ [{symbol}] Page loaded successfully")
            
            # Wait for the page to stabilize
            await browser_manager.page.wait_for_timeout(5000)
            
            # Find all dropdown buttons
            print(f"🔽 [{symbol}] Searching for dropdown buttons...")
            dropdown_button_selector = 'button.dropdown-toggle.btn-secondary.dropdown-toggle-no-caret'
            dropdown_buttons = await browser_manager.page.query_selector_all(dropdown_button_selector)
            
            if len(dropdown_buttons) < 3:
                raise ValueError(f"Expected at least 3 dropdown buttons, found {len(dropdown_buttons)}")
            
            # Select the third button (index 2)
            target_button = dropdown_buttons[2]
            button_text = await target_button.inner_text()
            button_text = button_text.strip()
            print(f"✅ [{symbol}] Selected third button with text: '{button_text}'")
            
            # Open the dropdown menu
            print(f"🔽 [{symbol}] Opening dropdown menu for '2567 - งบปี'...")
            try:
                await target_button.click()
                # Find the associated dropdown menu using aria-labelledby
                button_id = await target_button.get_attribute('id')
                dropdown_menu_locator = browser_manager.page.locator(f'.dropdown-menu[aria-labelledby="{button_id}"]')
                await dropdown_menu_locator.wait_for(state='visible', timeout=10000)
                
                # Get all dropdown items
                dropdown_items = await dropdown_menu_locator.locator('.dropdown-item').all()
                item_texts = [await item.inner_text() for item in dropdown_items]
                print(f"🔍 [{symbol}] Dropdown items for third button: {item_texts}")
                
                # Select "2567 - งบปี" in the dropdown
                target_option_found = False
                for item in dropdown_items:
                    item_text = await item.inner_text()
                    item_text = item_text.strip()
                    print(f"🔍 [{symbol}] Checking dropdown item: '{item_text}'")
                    if "2567 - งบปี" in item_text:
                        print(f"🎯 [{symbol}] Clicking option: '{item_text}'")
                        await item.click()
                        target_option_found = True
                        await browser_manager.page.wait_for_timeout(3000)
                        break
                
                if not target_option_found:
                    print(f"❌ [{symbol}] Available dropdown options: {item_texts}")
                    raise ValueError("Option '2567 - งบปี' not found in third button's dropdown")
                
            except Exception as e:
                print(f"❌ [{symbol}] Failed to select '2567 - งบปี' from dropdown: {e}")
                raise ValueError(f"Could not select '2567 - งบปี': {e}")
            
            # Click the search button with text "ค้นหา"
            print(f"🔍 [{symbol}] Clicking search button with text 'ค้นหา'...")
            search_button_selector = 'button.search-btn.bg-primary'
            try:
                await browser_manager.page.wait_for_selector(search_button_selector, state='visible', timeout=15000)
                search_button = await browser_manager.page.query_selector(search_button_selector)
                if not search_button:
                    # Fallback selector
                    search_button = await browser_manager.page.query_selector('button.search-btn')
                if not search_button:
                    raise ValueError("Search button not found")
                
                button_text = await search_button.inner_text()
                button_text = button_text.strip()
                print(f"🔍 [{symbol}] Found search button with text: '{button_text}'")
                
                if "ค้นหา" not in button_text:
                    raise ValueError(f"Search button text '{button_text}' does not match expected 'ค้นหา'")
                
                await search_button.click()
                print(f"✅ [{symbol}] Search button clicked")
                await browser_manager.page.wait_for_timeout(5000)
                
            except Exception as e:
                print(f"❌ [{symbol}] Failed to click search button: {e}")
                raise ValueError(f"Could not click search button: {e}")
            
            # Scroll to load all data
            await browser_manager.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await browser_manager.page.wait_for_timeout(2000)
            await browser_manager.page.evaluate('window.scrollTo(0, 0)')
            await browser_manager.page.wait_for_timeout(2000)
            
            print(f"📊 [{symbol}] Extracting financial data...")
            data_rows = await browser_manager.page.query_selector_all('div.row.py-2.ps-2.pe-3')
            if not data_rows:
                data_rows = await browser_manager.page.query_selector_all('div.row[class*="py-2"]')
            if not data_rows:
                raise ValueError("No data rows found on page after search")
            
            print(f"📊 [{symbol}] Found {len(data_rows)} data rows")
            metrics = []
            values = []
            
            for i, row in enumerate(data_rows):
                try:
                    metric_elements = await row.query_selector_all('div[class*="col"]')
                    if len(metric_elements) >= 2:
                        metric_text = await metric_elements[0].inner_text()
                        metric_text = metric_text.strip()
                        value_text = await metric_elements[1].inner_text()
                        value_text = value_text.strip()
                        
                        if (metric_text and 
                            not metric_text.startswith('สินทรัพย์:') and 
                            not metric_text.startswith('หนี้สิน:') and
                            not metric_text.startswith('ช่วงเวลา') and
                            metric_text != 'ค้นหา'):
                            metrics.append(metric_text)
                            values.append(value_text)
                            print(f"📈 [{symbol}] Row {i+1}: {metric_text} = {value_text}")
                
                except Exception as row_error:
                    print(f"⚠️ [{symbol}] Error processing row {i+1}: {row_error}")
                    continue
            
            if not metrics:
                raise ValueError("No valid financial metrics extracted from the page")
            
            print(f"📊 [{symbol}] Successfully extracted {len(metrics)} financial metrics")
            df = pd.DataFrame({
                'Symbol': symbol,
                'Financial_Metric': metrics,
                'Value': values
            })
            
            print(f"🧹 [{symbol}] Standardizing null values...")
            df = standardize_null_values(df)
            
            print(f"✅ [{symbol}] Successfully processed {len(df)} financial metrics")
            print(f"📋 [{symbol}] Sample data:")
            if len(df) > 0:
                print(df.head(5).to_string(index=False))
            
            null_counts = df.isna().sum()
            total_cells = len(df) * len(df.columns)
            total_nulls = null_counts.sum()
            print(f"📊 [{symbol}] Data quality statistics:")
            print(f"   Total records: {len(df)}")
            print(f"   Total cells: {total_cells}")
            print(f"   Total nulls: {total_nulls}")
            print(f"   Null percentage: {(total_nulls/total_cells)*100:.1f}%")
            
            return df
            
        except PlaywrightTimeoutError as e:
            print(f"⏰ [{symbol}] Timeout error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await browser_manager.restart_browser()
        except Exception as e:
            print(f"❌ [{symbol}] Error (Attempt {attempt + 1}/{max_retries}): {e}")
            import traceback
            print(f"🔍 [{symbol}] Full traceback:")
            traceback.print_exc()
            
            if "Connection closed" in str(e) or "Target page" in str(e) or "Element is not attached" in str(e):
                if attempt < max_retries - 1:
                    await browser_manager.restart_browser()
        
        if attempt < max_retries - 1:
            wait_time = random.uniform(3, 7)
            print(f"🔄 [{symbol}] Retrying in {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
    
    print(f"💥 [{symbol}] Failed to extract data after {max_retries} attempts")
    output_df = pd.DataFrame(columns=['Symbol', 'Financial_Metric', 'Value'])
    output_df['Symbol'] = [symbol]
    output_df['Financial_Metric'] = ['']
    output_df['Value'] = [pd.NA]
    return output_df
# ==============================================================================
# 5. MAIN SCRAPING FUNCTION WITH SPECIFIC BATCH FILE RESUME
# ==============================================================================
async def scrape_all_companies(csv_file_path, output_dir="financial_data", batch_size=50):
    """Main function to scrape financial data, resuming from specific batch file"""
    os.makedirs(output_dir, exist_ok=True)
    
    symbols = load_company_symbols(csv_file_path)
    if not symbols:
        print("❌ No symbols found. Exiting...")
        return
    
    # Define the specific batch file to resume from
    specific_batch_file = "financial_data_batch_350_20250616_173851.xlsx"
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    processed_symbols = set()
    last_batch_file = None
    start_index = 0
    
    # Load the specific batch file
    specific_batch_path = os.path.join(output_dir, specific_batch_file)
    if os.path.exists(specific_batch_path):
        try:
            batch_df = pd.read_excel(specific_batch_path)
            batch_symbols = batch_df['Symbol'].dropna().astype(str).str.strip().unique().tolist()
            processed_symbols.update(batch_symbols)
            last_batch_file = specific_batch_file
            start_index = int(specific_batch_file.split('_')[3])  # Extract index from filename (e.g., 350)
            print(f"🔄 Resuming from specified batch file: {specific_batch_file}")
            print(f"✅ Loaded {len(batch_symbols)} symbols from {specific_batch_file}")
            print(f"🏁 Starting from index: {start_index}")
        except Exception as e:
            logger.error(f"Failed to load specified batch file {specific_batch_file}: {e}")
            processed_symbols = set()
            last_batch_file = None
            start_index = 0
    else:
        logger.warning(f"Specified batch file {specific_batch_file} not found")
        
        # Fall back to checkpoint file if it exists
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    last_batch_file = checkpoint.get('last_batch_file', None)
                    processed_symbols = set(checkpoint.get('processed_symbols', []))
                    start_index = checkpoint.get('last_index', 0)
                    
                    # Verify last batch file from checkpoint
                    if last_batch_file and os.path.exists(os.path.join(output_dir, last_batch_file)):
                        try:
                            batch_df = pd.read_excel(os.path.join(output_dir, last_batch_file))
                            batch_symbols = batch_df['Symbol'].dropna().astype(str).str.strip().unique().tolist()
                            processed_symbols.update(batch_symbols)
                            print(f"🔄 Loaded {len(batch_symbols)} symbols from checkpoint batch file: {last_batch_file}")
                        except Exception as e:
                            logger.error(f"Failed to load checkpoint batch file {last_batch_file}: {e}")
                            processed_symbols = set(checkpoint.get('processed_symbols', []))
                    else:
                        logger.warning(f"Checkpoint batch file {last_batch_file} not found, using checkpoint processed symbols")
                    
                    print(f"🔄 Resuming with {len(processed_symbols)} symbols already processed")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                processed_symbols = set()
                start_index = 0
    
    # Validate start_index
    if start_index >= len(symbols):
        logger.warning(f"Start index {start_index} exceeds symbol list length {len(symbols)}. Starting from beginning.")
        start_index = 0
    
    browser_manager = BrowserManager()
    
    try:
        if not await browser_manager.initialize():
            print("❌ Failed to initialize browser. Exiting...")
            return
        
        all_data = []
        successful_extractions = len(processed_symbols)  # Count already processed symbols
        failed_companies = []
        
        print(f"\n🚀 Starting to scrape {len(symbols)} companies, {len(processed_symbols)} already processed...")
        
        # Adjust the starting point based on start_index
        for i, symbol in enumerate(symbols[start_index:], start=start_index + 1):
            if symbol in processed_symbols:
                print(f"⭐️ [{symbol}] Already processed, skipping...")
                continue
            
            print(f"\n--- Processing {i}/{len(symbols)}: {symbol} ---")
            
            try:
                company_data = await scrape_company_financial_data(browser_manager, symbol)
                
                if company_data is not None:
                    all_data.append(company_data)
                    successful_extractions += 1
                    processed_symbols.add(symbol)
                    print(f"✅ [{symbol}] Added to results")
                else:
                    failed_companies.append(symbol)
                    print(f"❌ [{symbol}] Failed to extract data")
            
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {e}")
                failed_companies.append(symbol)
                
                if "Connection closed" in str(e) or "Browser" in str(e) or "Target page" in str(e):
                    logger.info("Attempting to restart browser due to critical error...")
                    await browser_manager.restart_browser()
            
            # Save intermediate results and update checkpoint
            if (len(all_data) >= batch_size or i == len(symbols)) and all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"financial_data_batch_{i}_{timestamp}.xlsx"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        combined_df.to_excel(writer, index=False, sheet_name='Financial_Data', na_rep='NA')
                    print(f"💾 Saved batch results to: {filepath}")
                    
                    # Update checkpoint
                    checkpoint = {
                        'last_batch_file': filename,
                        'processed_symbols': list(processed_symbols),
                        'last_index': i
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f, indent=2)
                    print(f"✅ Updated checkpoint: last_batch_file={filename}, processed={len(processed_symbols)} symbols")
                    
                    # Clear all_data to free memory
                    all_data = []
                
                except Exception as e:
                    logger.error(f"Failed to save batch file: {e}")
                
                print(f"📊 Progress: {successful_extractions}/{i} successful extractions")
            
            # Random delay between requests
            delay = random.uniform(2, 5)
            print(f"⏳ Waiting {delay:.1f} seconds...")
            await asyncio.sleep(delay)
    
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user...")
        if all_data:
            try:
                combined_df = pd.concat(all_data, ignore_index=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"financial_data_interrupted_{i}_{timestamp}.xlsx"
                filepath = os.path.join(output_dir, filename)
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    combined_df.to_excel(writer, index=False, sheet_name='Financial_Data', na_rep='NA')
                print(f"💾 Saved interrupted batch to: {filepath}")
                
                # Update checkpoint on interruption
                checkpoint = {
                    'last_batch_file': filename,
                    'processed_symbols': list(processed_symbols),
                    'last_index': i
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"✅ Updated checkpoint: last_batch_file={filename}, processed={len(processed_symbols)} symbols")
            except Exception as e:
                logger.error(f"Failed to save interrupted data: {e}")
        print(f"📊 Progress at interruption: {successful_extractions}/{i} successful extractions")
    
    except Exception as e:
        logger.error(f"Critical error in main scraping function: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔧 Cleaning up browser resources...")
        try:
            await browser_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during browser cleanup: {e}")
        
        print(f"\n🎉 --- SCRAPING COMPLETE --- 🎉")
        print(f"✅ Successfully extracted: {successful_extractions}/{len(symbols)} companies")
        print(f"❌ Failed extractions: {len(failed_companies)}")
        
        if failed_companies:
            try:
                failed_df = pd.DataFrame({'Failed_Symbols': failed_companies})
                failed_file = os.path.join(output_dir, f"failed_companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                with pd.ExcelWriter(failed_file, engine='openpyxl') as writer:
                    failed_df.to_excel(writer, index=False)
                print(f"📝 Failed companies saved to: {failed_file}")
            except Exception as e:
                logger.error(f"Failed to save failed companies list: {e}")
        
        if all_data:
            try:
                final_df = pd.concat(all_data, ignore_index=True)
                final_file = os.path.join(output_dir, f"final_financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                with pd.ExcelWriter(final_file, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='Financial_Data', na_rep='NA')
                print(f"📊 Final data saved to: {final_file}")
                print(f"📈 Total records: {len(final_df)}")
                
                # Update checkpoint with final state
                checkpoint = {
                    'last_batch_file': final_file,
                    'processed_symbols': list(processed_symbols),
                    'last_index': i
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"✅ Updated checkpoint: last_batch_file={final_file}, processed={len(processed_symbols)} symbols")
            except Exception as e:
                logger.error(f"Failed to save final data: {e}")

# ==============================================================================
# 6. TEST FUNCTION FOR 3BBIF
# ==============================================================================
async def test_single_company(symbol="3BBIF"):
    """Test scraping for 3BBIF with updated div-based structure"""
    print(f"🧪 Testing scraping for company: {symbol}")
    
    browser_manager = BrowserManager()
    
    try:
        if not await browser_manager.initialize():
            print("❌ Failed to initialize browser for testing")
            return
        
        result = await scrape_company_financial_data(browser_manager, symbol)
        
        if result is not None and len(result) > 0:
            print(f"\n🎉 SUCCESS! Extracted data for {symbol}")
            print(f"📊 Shape: {result.shape}")
            print(f"🏷️ Columns: {result.columns.tolist()}")
            print("\n📋 Full Data Preview:")
            print(result.to_string(index=False))
            
            os.makedirs("test_output", exist_ok=True)
            test_file = f"test_output/test_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            try:
                with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
                    result.to_excel(writer, index=False, sheet_name='Financial_Data', na_rep='NA')
                print(f"\n💾 Test result saved to: {test_file}")
            except Exception as e:
                logger.error(f"Failed to save test result: {e}")
            
            print(f"\n📊 Data Quality Statistics:")
            null_count = result['Value'].isna().sum()
            non_null_count = result['Value'].notna().sum()
            print(f"   Value column: {non_null_count} values, {null_count} NAs")
            print(f"   Total metrics: {len(result)}")
            
        else:
            print(f"\n❌ FAILED to extract data for {symbol}")
            
    finally:
        await browser_manager.cleanup()

# ==============================================================================
# 7. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Running test with 3BBIF...")
    asyncio.run(test_single_company("3BBIF"))
    
    CSV_FILE_PATH = "C:/Users/Zixun/OneDrive - Nanyang Technological University/vs_python/Vekin/triangle-app/listedCompanies_en_US.csv"
    OUTPUT_DIRECTORY = "financial_data"
    BATCH_SIZE = 50
    
    if os.path.exists(CSV_FILE_PATH):
        asyncio.run(scrape_all_companies(
            csv_file_path=CSV_FILE_PATH,
            output_dir=OUTPUT_DIRECTORY,
            batch_size=BATCH_SIZE
        ))
    else:
        print(f"❌ CSV file not found: {CSV_FILE_PATH}")