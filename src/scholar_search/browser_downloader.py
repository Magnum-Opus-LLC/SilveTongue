"""
Browser-based PDF downloader using Playwright (Puppeteer alternative for Python).
Handles complex pages that require JavaScript rendering or authentication.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright not installed. Browser-based downloading disabled. Install with: pip install playwright && playwright install")


class BrowserDownloader:
    """
    Downloads PDFs using browser automation for sites that require JavaScript.
    """

    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize browser downloader.

        Args:
            headless: Run browser in headless mode
            timeout: Default timeout in milliseconds
        """
        self.headless = headless
        self.timeout = timeout
        self.playwright = None
        self.browser = None

    async def __aenter__(self):
        """Async context manager entry."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        logger.info("Browser launched")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser closed")

    async def download_pdf_from_page(self, url: str, output_path: Path, strategy: str = 'auto') -> bool:
        """
        Download PDF from a page using browser automation.

        Args:
            url: Page URL to visit
            output_path: Where to save the PDF
            strategy: Download strategy:
                - 'auto': Auto-detect PDF download button/link
                - 'direct': URL is direct PDF, just download
                - 'button': Click download button
                - 'print': Use browser print to PDF

        Returns:
            True if successful, False otherwise
        """
        if not self.browser:
            logger.error("Browser not initialized. Use async context manager.")
            return False

        logger.info(f"Downloading from {url} using strategy '{strategy}'")

        try:
            page = await self.browser.new_page()

            # Navigate to page
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            logger.info(f"  Page loaded: {await page.title()}")

            if strategy == 'direct':
                # URL points directly to PDF
                # Playwright will auto-download if Content-Type is application/pdf
                async with page.expect_download() as download_info:
                    await page.goto(url)
                download = await download_info.value
                await download.save_as(output_path)
                logger.info(f"  ✓ Direct PDF downloaded")
                return True

            elif strategy == 'button':
                # Look for and click download button
                download_selectors = [
                    'a:has-text("Download PDF")',
                    'button:has-text("Download PDF")',
                    'a:has-text("Download")',
                    'a[href*=".pdf"]',
                    '.download-pdf',
                    '#download-pdf',
                ]

                for selector in download_selectors:
                    try:
                        element = await page.wait_for_selector(selector, timeout=5000)
                        if element:
                            logger.info(
                                f"  Found download element: {selector}")

                            # Set up download handler
                            async with page.expect_download() as download_info:
                                await element.click()

                            download = await download_info.value
                            await download.save_as(output_path)
                            logger.info(f"  ✓ PDF downloaded via button")
                            return True
                    except PlaywrightTimeout:
                        continue

                logger.warning("  No download button found")
                return False

            elif strategy == 'print':
                # Use browser print to PDF
                await page.pdf(path=str(output_path))
                logger.info(f"  ✓ PDF generated via print")
                return True

            elif strategy == 'auto':
                # Auto-detect: try button first, then print
                try:
                    return await self.download_pdf_from_page(url, output_path, 'button')
                except Exception:
                    logger.info("  Button strategy failed, trying print")
                    return await self.download_pdf_from_page(url, output_path, 'print')

            else:
                logger.error(f"Unknown strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Browser download failed: {e}")
            return False

        finally:
            await page.close()

    async def extract_pdf_url_with_browser(self, landing_page: str) -> Optional[str]:
        """
        Use browser to navigate to landing page and extract actual PDF URL.
        Useful for pages with redirects or JavaScript-generated links.

        Args:
            landing_page: Page to visit

        Returns:
            PDF URL if found, None otherwise
        """
        if not self.browser:
            logger.error("Browser not initialized")
            return None

        logger.info(f"Extracting PDF URL from: {landing_page}")

        try:
            page = await self.browser.new_page()
            await page.goto(landing_page, timeout=self.timeout, wait_until='networkidle')

            # Look for PDF links
            pdf_selectors = [
                'a[href*=".pdf"]',
                'a:has-text("PDF")',
                'a:has-text("Download")',
                'meta[name="citation_pdf_url"]',
            ]

            for selector in pdf_selectors:
                try:
                    if selector.startswith('meta'):
                        # Meta tag
                        element = await page.wait_for_selector(selector, timeout=3000)
                        if element:
                            pdf_url = await element.get_attribute('content')
                            logger.info(f"  Found PDF URL in meta: {pdf_url}")
                            return pdf_url
                    else:
                        # Link
                        element = await page.wait_for_selector(selector, timeout=3000)
                        if element:
                            pdf_url = await element.get_attribute('href')
                            if pdf_url and '.pdf' in pdf_url:
                                logger.info(f"  Found PDF URL: {pdf_url}")
                                return pdf_url
                except PlaywrightTimeout:
                    continue

            logger.warning("  No PDF URL found")
            return None

        except Exception as e:
            logger.error(f"Failed to extract PDF URL: {e}")
            return None

        finally:
            await page.close()


async def download_with_browser(url: str, output_path: Path, strategy: str = 'auto') -> bool:
    """
    Convenience function to download a single PDF with browser.

    Args:
        url: URL to download from
        output_path: Where to save PDF
        strategy: Download strategy (auto, direct, button, print)

    Returns:
        True if successful
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright not installed")
        return False

    async with BrowserDownloader() as downloader:
        return await downloader.download_pdf_from_page(url, output_path, strategy)


def download_with_browser_sync(url: str, output_path: Path, strategy: str = 'auto') -> bool:
    """
    Synchronous wrapper for browser download.
    """
    return asyncio.run(download_with_browser(url, output_path, strategy))


if __name__ == '__main__':
    # Test browser downloader
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python browser_downloader.py <url> <output_path> [strategy]")
        sys.exit(1)

    url = sys.argv[1]
    output = Path(sys.argv[2])
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'auto'

    success = download_with_browser_sync(url, output, strategy)
    print(f"{'✓' if success else '✗'} Download {'succeeded' if success else 'failed'}")
