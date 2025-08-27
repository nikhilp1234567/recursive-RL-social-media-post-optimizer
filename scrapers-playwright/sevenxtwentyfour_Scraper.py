import time
from playwright.sync_api import sync_playwright, TimeoutError

def scrape_7x24_emails():
    """
    Launches a browser, navigates to the 7x24 Exchange member list,
    applies filters, scrapes emails for the filtered companies,
    and saves them to a text file.
    """
    # --- Configuration ---
    TARGET_URL = "https://www.7x24exchange.org/membership/member-list/"
    OUTPUT_FILE = "7x24emails.txt"

    # List of categories to select based on the actual HTML structure
    CATEGORIES_TO_SELECT = [
        "Architecture/Design", "Cloud", "Construction", "Consulting Services",
        "Data Management/Storage", "Equipment Rentals", "Financial Services",
        "Personnel", "Safety/Fire Prevention", "Telecommunications",
        "Artificial Intelligence", "Business/Disaster Recovery", "Colocation",
        "Commissioning", "Control Systems", "Data Center (Physical)",
        "Engineering Services", "Hardware/Software", "Power Generation/Distribution",
        "Real Estate/Property Management", "Technology Services"
    ]

    all_emails = []

    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=False, slow_mo=50) # headless=False is useful for debugging
        page = browser.new_page()

        try:
            print(f"Navigating to {TARGET_URL}...")
            page.goto(TARGET_URL, timeout=60000)

            print("Waiting for the page to load...")
            # Wait for the main content to be visible
            page.wait_for_selector('div.container', timeout=30000)
            
            print("Waiting for manual filter application...")
            # Wait 10 seconds for manual filter application in browser
            time.sleep(10)
            print("Proceeding with scraping...")

            # The companies are actually in an iframe! Let's switch to it
            print("Switching to member list iframe...")
            iframe = page.frame_locator('iframe[src*="memberlistfeed.php"]').first
            if not iframe:
                print("Could not find member list iframe!")
                return
            
            print("Waiting for companies to load in iframe...")
            # Wait for the accordian elements to appear in the iframe
            iframe.locator('div.accordian').first.wait_for(timeout=30000)
            
            print("Scraping company details...")
            # Get all the company containers from the iframe
            company_accordians = iframe.locator('div.accordian').all()
            print(f"Found {len(company_accordians)} companies matching the filter.")

            for i, accordian in enumerate(company_accordians):
                try:
                    # Click the accordian to expand it and show details
                    accordian.click()
                    
                    # A brief pause to allow for the dropdown animation
                    page.wait_for_timeout(500)

                    # Find the email in the details section using the actual HTML structure
                    # The email is in a div with class "dtl" and follows the pattern: <b>General Contact Email:</b><br>email@domain.com
                    email_locator = accordian.locator('div.dtl')
                    
                    if email_locator.count() > 0:
                        # Get the text content and extract email
                        details_text = email_locator.inner_text()
                        if 'General Contact Email:' in details_text:
                            # Extract email from the text
                            email_start = details_text.find('General Contact Email:') + len('General Contact Email:')
                            email_end = details_text.find('\n', email_start)
                            if email_end == -1:
                                email_end = len(details_text)
                            
                            email = details_text[email_start:email_end].strip()
                            if email and '@' in email and email not in all_emails:
                                # Get company name from the widget-header link
                                company_name_elem = accordian.locator('div.widget-header a')
                                if company_name_elem.count() > 0:
                                    company_name = company_name_elem.inner_text()
                                else:
                                    company_name = f"Company {i+1}"
                                
                                print(f"  ({i+1}/{len(company_accordians)}) Found email for {company_name}: {email}")
                                all_emails.append(email)
                    
                    # Collapse the accordian before moving to the next
                    accordian.click()

                except TimeoutError:
                    print(f"  - Timeout while processing an entry, skipping.")
                except Exception as e:
                    print(f"  - An error occurred on an entry: {e}")

        except TimeoutError:
            print("A timeout occurred. The page might be slow to load or a selector might not be found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Closing browser...")
            browser.close()

    # --- Save the results ---
    if all_emails:
        print(f"\nScraping complete. Found {len(all_emails)} unique emails.")
        with open(OUTPUT_FILE, "w") as f:
            for email in all_emails:
                f.write(email + "\n")
        print(f"Successfully saved emails to {OUTPUT_FILE}")
    else:
        print("Scraping finished, but no emails were found. The selectors might need adjustment if the site has changed.")

if __name__ == "__main__":
    scrape_7x24_emails()