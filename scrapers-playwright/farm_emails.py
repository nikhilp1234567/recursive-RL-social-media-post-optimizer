import asyncio
from playwright.async_api import async_playwright
import re

async def main():
    """
    Scrapes email addresses from farm URLs listed in farm_urls.txt
    and saves them to farm_url_emails.txt

    Input file lines:
      https://example.com/org/public-profile/123 , Farm Name

    Output file lines:
      email@example.com , Farm Name
    """
    # Read the farm URLs and names from the file
    try:
        with open('farm_urls.txt', 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        print(f"Found {len(raw_lines)} entries to process")
    except FileNotFoundError:
        print("Error: farm_urls.txt file not found")
        return
    except Exception as e:
        print(f"Error reading farm_urls.txt: {e}")
        return

    # Pre-parse into (url, name)
    entries = []
    for line in raw_lines:
        # Split on first comma only to preserve commas inside names, if any
        parts = line.split(',', 1)
        url = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        entries.append((url, name))

    # Open output file for writing emails and names
    with open('farm_url_emails.txt', 'w', encoding='utf-8') as output_file:
        print("Starting Playwright session to scrape email addresses...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                for i, (url, name) in enumerate(entries, 1):
                    display_name = f" ({name})" if name else ""
                    print(f"Processing {i}/{len(entries)}: {url}{display_name}")
                    
                    try:
                        # Navigate to the URL
                        await page.goto(url, timeout=30000)
                        
                        # Wait for the page to load
                        await page.wait_for_load_state('networkidle')
                        
                        # Look for email elements using the class structure from the image
                        email_elements = page.locator('.views-field-email a[href^="mailto:"]')
                        
                        if await email_elements.count() > 0:
                            # Get the first email link found
                            email_link = email_elements.first
                            email_href = await email_link.get_attribute('href')
                            
                            if email_href and email_href.startswith('mailto:'):
                                # Extract email from mailto: link
                                email = email_href.replace('mailto:', '').strip()
                                print(f"  Found email: {email}")
                                
                                # Write "email , name" to file
                                output_file.write(f"{email} , {name}\n")
                                output_file.flush()  # Ensure it's written immediately
                            else:
                                print(f"  No valid email found in href: {email_href}")
                                output_file.write(f"No email found , {name}\n")
                        else:
                            print(f"  No email elements found on this page")
                            output_file.write(f"No email found , {name}\n")
                            
                    except Exception as e:
                        print(f"  Error processing {url}: {e}")
                        output_file.write(f"Error processing URL , {name}\n")
                    
                    # Small delay between requests to be respectful
                    await asyncio.sleep(1)
                
                print(f"\nScraping complete! All emails have been saved to farm_url_emails.txt")

            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                await browser.close()

if __name__ == "__main__":
    asyncio.run(main())