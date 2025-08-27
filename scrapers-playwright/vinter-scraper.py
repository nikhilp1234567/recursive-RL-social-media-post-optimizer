import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    """
    This is the main asynchronous function that runs the Playwright script.
    It navigates to the specified URL, scrapes vintner data, and saves it to a file.
    """
    # URL of the page to scrape
    url = "https://napavalley.wine/makers/vintners"
    
    # Name of the output file
    output_file = "vintners.txt"

    # Use a file context manager to ensure the file is closed properly
    with open(output_file, 'w', encoding='utf-8') as f:
        print(f"Starting Playwright session to scrape data from {url}...")
        
        # Start the Playwright async context
        async with async_playwright() as p:
            # Launch the browser in headless mode (no GUI visible)
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Navigate to the URL
                await page.goto(url)
                print("Page loaded successfully. Searching for vintner panels...")

                # Await a brief moment to ensure all elements are rendered
                await page.wait_for_selector(".vintner-list-item", state="visible", timeout=30000)

                # Find all the vintner panels on the page. 
                # .vintner-list-item is the class name for each individual panel.
                vintner_panels = await page.locator(".vintner-list-item").all()
                print(f"Found {len(vintner_panels)} vintner panels. Starting to scrape...")

                # Loop through each panel
                for i, panel in enumerate(vintner_panels):
                    # Playwright handles the hover action seamlessly, which ensures the text is visible
                    # even if it's initially hidden.
                    await panel.hover()

                    # Find the name and company elements within the current panel
                    try:
                        # Based on the HTML structure, let's try different selectors
                        # First, let's get all text content to see what's available
                        panel_text = await panel.text_content()
                        print(f"Panel {i+1} text content: {panel_text}")
                        
                        # Try to find the name - it might be in an alt attribute or different element
                        name_element = panel.locator("img")
                        if await name_element.count() > 0:
                            alt_text = await name_element.get_attribute("alt")
                            if alt_text and "Maker Story -" in alt_text:
                                name = alt_text.replace("Maker Story - ", "").strip()
                            else:
                                name = "Name not found"
                        else:
                            name = "Name not found"
                        
                        # For company, we might need to look at the link text or other elements
                        company_element = panel.locator("a")
                        if await company_element.count() > 0:
                            company = await company_element.get_attribute("href")
                            if company:
                                # Extract company info from URL or try to find it elsewhere
                                company = company.split("/")[-1].replace("-", " ").title()
                            else:
                                company = "Company not found"
                        else:
                            company = "Company not found"

                        # Format the output string
                        output_line = f"Name: {name}, Company: {company}"
                        print(f"Scraped data for panel {i+1}: {output_line}")
                        
                        # Write the data to the file, followed by a new line
                        f.write(output_line + "\n")
                    except Exception as e:
                        print(f"Could not find name/company for panel {i+1}. Skipping. Error: {e}")
                
                print(f"\nScraping complete! All data has been saved to {output_file}.")

            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                # Close the browser session
                await browser.close()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
