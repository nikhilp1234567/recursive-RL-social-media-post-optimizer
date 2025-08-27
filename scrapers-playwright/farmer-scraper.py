import asyncio
import re
import os
from playwright.async_api import async_playwright

async def main():
    """
    This script scrapes a website to find organizations with "farm" in their name,
    and saves their corresponding URLs to a single text file, each on a new line.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) # Set headless=True to run in the background
        page = await browser.new_page()

        # The URL to scrape
        url = "https://www.farmgarden.org.uk/about-us/our-members"
        await page.goto(url)

        print(f"Navigated to {url}")
        # Add a 5 second wait and logging
        print("Waiting 5 seconds to ensure the page is fully loaded...")
        await page.wait_for_timeout(10000)
        print("Wait complete. Proceeding with scraping.")

        # The selector for the links in the table based on the provided screenshot
        # This targets the <a> tag inside the <td> with the specified class.
        member_links_selector = "td.views-field-display-name a"

        # Wait for the table to be visible to ensure the content has loaded
        await page.wait_for_selector(member_links_selector, timeout=30000)
        print("Member table is visible.")

        # Get all the links that match the selector
        member_locators = page.locator(member_links_selector)
        count = await member_locators.count()
        print(f"Found {count} total member links on the page.")

        # We need to get the hrefs and text first, because navigating will detach the locators
        member_details = []
        for i in range(count):
            locator = member_locators.nth(i)
            text = await locator.inner_text()
            href = await locator.get_attribute("href")
            member_details.append({"text": text, "href": href})

        # Define the single output file
        output_file = "farm_urls.txt"

        # Open the single file to write all URLs
        with open(output_file, "w", encoding="utf-8") as f:
            print(f"Saving URLs to {output_file}")
            # Iterate through the collected details
            for member in member_details:
                member_name = member["text"]
                member_href = member["href"]

                # Check if "farm" is in the name (case-insensitive)
                if "farm" in member_name.lower():
                    print(f"Processing: {member_name}")

                    # Create a full URL if the href is relative
                    full_url = member_href
                    if not full_url.startswith("http"):
                        full_url = f"https://www.farmgarden.org.uk{full_url}"

                    try:
                        # Save the full URL to the file, followed by a newline
                        f.write(f"{full_url} , {member_name}\n")
                        print(f"  -> Successfully saved URL for {member_name}")

                    except Exception as e:
                        print(f"  -> Could not save URL for {member_name}: {e}")
            
        print(f"\nScraping complete. All URLs saved to {output_file}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
